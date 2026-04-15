#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "optimizers.h"

void zero_grad(Optimizer* optim){

    for(int i=0; i<optim->p_count; i++){
        Tensor* p = optim->params[i];

        if(p->grad) memset(p->grad, 0, sizeof(double) * p->size);
    }
}

void clip_grad_norm(Optimizer* optim, double max_norm){


    double total_norm_sq = 0.0;
    double eps = 1e-8;

    for(int i=0; i<optim->p_count; i++){
        
        Tensor* p = optim->params[i];
        if(!p) continue;
        
        for(int j=0; j<p->size; j++) total_norm_sq += p->grad[j] * p->grad[j];
    
    }

    total_norm_sq = sqrt(total_norm_sq);

    if(total_norm_sq > max_norm){
        
        double scale = max_norm / (total_norm_sq + eps);

        for(int i=0; i<optim->p_count; i++){

            Tensor* p = optim->params[i];
            
            if(!p) continue;

            for(int j=0; j<p->size; j++){
                p->grad[j] *= scale;
            }
        }
    }
}


void sync_weights(Optimizer* optim){
    
    int B = optim->batch_size;

    for(int k=0; k<optim->p_count; k++){
        Tensor* p = optim->params[k];

        int N = p->size / B;


        for(int i=0; i < N; i++){
            
            // summing up all gradients of w[i] across batches
            //g_sum actually the mean but we divide the sum by the number of distinct w in training loop because our loss functions will do that.
            
            double g_sum = 0.0;
            for(int lane=0; lane < B; lane++){
                g_sum += p->grad[lane * N + i];
            }

            p->grad[i] = g_sum;
        }
    }
}

void _sgd_momentum(Optimizer* optim){
    
    sync_weights(optim);
    
    int B = optim->batch_size;

    for(int k=0; k<optim->p_count; k++){
        
        Tensor* p = optim->params[k];
        double* vel = optim->velocity[k];

        int N = p->size / B;

        for(int i=0; i<N; i++){

            double g_sum = p->grad[i]; // summed gradient from sync weights

            vel[i] = (optim->beta * vel[i]) + (1 - optim->beta) * g_sum;
            
            double w_final = p->data[i] - optim->lr * vel[i];
            
            for(int lane=0; lane < B; lane++){
                p->data[lane * N + i] = w_final;

                // let's do the zero_grad function's job here just to be safe, because its C fellas!
                p->grad[lane * N + i] = 0.0;
            }
        }
    }
}

Optimizer* sgd(Tensor** params, int count, int batch_size, double lr, double beta){
    
    Optimizer* optim = (Optimizer*)malloc(sizeof(Optimizer));
 
    optim->params = params;
    optim->p_count = count;
    optim->batch_size = batch_size;
    optim->lr = lr;
    optim->beta = beta;
    optim->step = _sgd_momentum;

    optim->velocity = (double**)malloc(count * sizeof(double*));
    for(int i=0; i<count; i++){
        
        int size = params[i]->size;
        
        int N = size / batch_size;
        optim->velocity[i] = (double*)calloc(N, sizeof(double));
    }

    return optim;

}