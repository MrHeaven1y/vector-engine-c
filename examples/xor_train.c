#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "graph.h"
#include "ops.h"
#include "layers.h"
#include "optimizers.h"
#include "stensor.h"
#include "io.h"

int main(){
    srand(time(NULL));

    printf("--- Vector Engine: XOR Training ---\n\n");

    GraphContext* ctx = init_graph(10000, 1000, 100);
    Model* model = init_model(100);

    int batch_size = 4;

    double x0_data[] = {0.0, 0.0, 1.0, 1.0};
    double x1_data[] = {0.0, 1.0, 0.0, 1.0};
    double y_data[] = {0.0, 1.0, 1.0, 0.0};

    Tensor* in_tensors[2];
    
    in_tensors[0] = Input(ctx, x0_data, batch_size);
    in_tensors[1] = Input(ctx, x1_data, batch_size);
    Tensor* target = Input(ctx, y_data, batch_size);

    DenseLayer* d1 = _dense_layer(ctx, model, 2, 4, batch_size);
    DenseLayer* d2 = _dense_layer(ctx, model, 4, 1, batch_size);

    Optimizer* opt = sgd(model->params, model->count, batch_size, 0.5, 0.9);

    int epochs = 6000;
    for(int epoch = 0; epoch <= epochs; epoch++){

        Tensor** h1 = Dense(ctx, d1, in_tensors);

        Tensor* h1_act[4];
        for(int i = 0; i < 4; i++){
            h1_act[i] = Tanh(ctx, h1[i]);
        }

        Tensor** out = Dense(ctx, d2, h1_act);
        Tensor* pred = Sigmoid(ctx, out[0]);

        Tensor* loss = Mse(ctx, target, pred);

        backward(ctx, loss);

        opt->step(opt);

        if (epoch % 100 == 0){
            printf("Epoch %d | MSE Loss: %f\n", epoch, loss->data[0]);
        }

        reset_tape(ctx);

        free(h1);
        free(out);
    }

    printf("\n--- True Inference Test (Batch Size = 1) ---\n");

    resize_model_batch(model, 1);

    double test_x0[] = {1.0};
    double test_x1[] = {0.0};

    Tensor* single_input[2];
    single_input[0] = Input(ctx, test_x0, 1);
    single_input[1] = Input(ctx, test_x1, 1);

    Tensor** h1_eval = Dense(ctx, d1, single_input);
    
    Tensor* h1_act_eval[4];
    for(int i=0; i<4; i++){
        h1_act_eval[i] = Tanh(ctx, h1_eval[i]);
    }

    Tensor** out_eval = Dense(ctx, d2, h1_act_eval);
    Tensor* pred_eval = Sigmoid(ctx, out_eval[0]);

    printf("Input: (1.0, 0.0) | Predicted: %.4f\n", pred_eval->data[0]);

    printf("\nSaving resized inference model to xor_model.json...\n");
    save_model(pred_eval, 1000, "xor_model.json");
    printf("Save Complete!\n");

    free(h1_eval);
    free(out_eval);

    reset_graph(ctx);
    free(ctx);
    free(model->params);
    free(model);

    return 0;

}