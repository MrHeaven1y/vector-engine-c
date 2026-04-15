#include "../include/ops.h"
#include "../include/layers.h"
#include "../include/optimizers.h"
#include "../include/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void shuffle_indices(int* indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
}

// --- RESIZE HELPER (Fixes the Inference Bug) ---
void resize_model_batch(Model* m, int new_batch_size) {
    for(int i=0; i < m->count; i++) {
        Tensor* t = m->params[i];
        
        // 1. Keep the first weight value (the learned weight)
        double learned_val = t->data[0];

        // 2. Reallocate data
        free(t->data);
        if(t->grad) free(t->grad);

        t->size = new_batch_size;
        t->data = malloc(sizeof(double) * new_batch_size);
        t->grad = calloc(new_batch_size, sizeof(double)); // grad is 0

        // 3. Broadcast the learned value to the new size
        for(int j=0; j < new_batch_size; j++) {
            t->data[j] = learned_val;
        }
    }
}

// --- THE MLP FIT FUNCTION ---
void fit_mlp(GraphContext* ctx, Model* m, Optimizer* opt, 
             DenseLayer* hidden, DenseLayer* output,
             double* x_train, double* y_train, 
             int total_samples, int input_dim, int classes,
             int epochs, int batch_size) 
{
    printf("Training MLP: %d samples | Batch %d | %d Hidden\n", total_samples, batch_size, hidden->output_count);

    // Buffers
    double** feature_cols = malloc(sizeof(double*) * input_dim);
    for(int i=0; i<input_dim; i++) feature_cols[i] = malloc(sizeof(double) * batch_size);

    double** target_cols = malloc(sizeof(double*) * classes);
    for(int i=0; i<classes; i++) target_cols[i] = malloc(sizeof(double) * batch_size);

    int* indices = malloc(sizeof(int) * total_samples);
    for(int i=0; i<total_samples; i++) indices[i] = i;

    for(int epoch = 0; epoch < epochs; epoch++) {
        shuffle_indices(indices, total_samples);
        double epoch_loss = 0.0;
        int batch_count = 0;

        for(int start = 0; start < total_samples; start += batch_size) {
            
            int B = batch_size; 
            // Handle last partial batch (skip it for simplicity in SIMD, or pad it)
            if (start + B > total_samples) break; 

            // --- DATA PREP ---
            for(int c=0; c<classes; c++) 
                for(int b=0; b<B; b++) target_cols[c][b] = 0.0;

            for(int b = 0; b < B; b++) {
                int real_idx = indices[start + b];
                for(int f = 0; f < input_dim; f++) 
                    feature_cols[f][b] = x_train[real_idx * input_dim + f];
                
                int class_idx = (int)y_train[real_idx];
                if(class_idx >= 0 && class_idx < classes) target_cols[class_idx][b] = 1.0;
            }

            // --- GRAPH BUILDING ---
            
            // 1. Inputs
            Tensor** inputs = malloc(sizeof(Tensor*) * input_dim);
            for(int f=0; f<input_dim; f++) inputs[f] = Input(ctx, feature_cols[f], B);

            Tensor** targets = malloc(sizeof(Tensor*) * classes);
            for(int c=0; c<classes; c++) targets[c] = Input(ctx, target_cols[c], B);

            // 2. Forward
            Tensor** hidden_out = DenseForward(ctx, hidden, inputs);
            
            Tensor** hidden_act = malloc(sizeof(Tensor*) * hidden->output_count);
            for(int i=0; i<hidden->output_count; i++) hidden_act[i] = Relu(ctx, hidden_out[i]);

            Tensor** logits = DenseForward(ctx, output, hidden_act);

            // 3. Loss (FIXED: Using the Optimized Implicit Function)
            Tensor* loss_node = NULL;
            
            if (classes > 1) {
                // We create a "Virtual" tensor that aggregates all logits/targets for the loss op
                // But since our Softmax_CE takes (Tensor* target, Tensor* pred), and your 'logits' is Tensor**,
                // We need to pick the "active" logit logic or just loop.
                
                // SIMPLER APPROACH FOR NOW: 
                // We use the Softmax_CE on each class column? No, Softmax implies competition.
                // We must use the manual build OR update Softmax_CE to take Tensor**.
                
                // Let's stick to your manual build for now BUT fix the stability
                Tensor** probs = build_softmax_graph(ctx, logits, classes);
                loss_node = build_categorical_ce(ctx, probs, targets, classes);
                free(probs); 
            } else {
                Tensor* prob = Sigmoid(ctx, logits[0]);
                loss_node = Binary_CE(ctx, targets[0], prob);
            }
            
            // 4. Gradient Scale
            if(!loss_node->grad) loss_node->grad = calloc((size_t)B, sizeof(double));
            
            double batch_loss_val = 0.0;
            for(int i=0; i<B; i++) {
                batch_loss_val -= loss_node->data[i]; 
                loss_node->grad[i] = -1.0 / B; 
            }

            // --- OPTIMIZE ---
            backward(ctx, loss_node); 
            sync_weights(opt, B);
            clip_grad_norm(opt, 5.0);
            opt->step(opt);

            // --- CLEANUP ---
            epoch_loss += batch_loss_val;
            batch_count++;
            
            reset_computation(ctx);
            free(inputs);
            free(targets);
            free(hidden_out);
            free(hidden_act);
            free(logits);
        }

        if((epoch+1) % 50 == 0) {
            printf("Epoch %d | Loss: %.6f\n", epoch+1, epoch_loss / batch_count / batch_size); 
            fflush(stdout); 
        }
    }

    // --- CRITICAL FIX: Resize Model for Inference ---
    printf("Training Done. Resizing model to Batch Size 1 for Inference...\n");
    resize_model_batch(m, 1);

    // Final Cleanup
    for(int i=0; i<input_dim; i++) free(feature_cols[i]);
    free(feature_cols);
    for(int i=0; i<classes; i++) free(target_cols[i]);
    free(target_cols);
    free(indices);
}