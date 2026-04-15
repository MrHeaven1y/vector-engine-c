#include<stdlib.h>

#include "layers.h"
#include "ops.h"

Model* init_model(int capacity){
    Model* m = (Model*)malloc(sizeof(Model));
    m->cap = capacity;
    m->count = 0;

    m->params = (Tensor**)malloc(sizeof(Tensor*) * m->cap);

    return m;
}


void register_param(Model* m, Tensor* param){
    
    if(m->count >= m->cap){
        m->cap *= 2;
        m->params = (Tensor**)realloc(m->params, sizeof(Tensor*) * m->cap);
    }

    m->params[m->count++] = param; 
}

void resize_model_batch(Model* m, int new_batch_size){
    for(int i=0; i < m->count; i++){
        Tensor* t = m->params[i];
        
        double learned_val = t->data[0];

        free(t->data);
        if(t->grad) free(t->grad);

        t->size = new_batch_size;
        t->data = (double*)malloc(sizeof(double) * new_batch_size);
        t->grad = (double*)calloc(new_batch_size, sizeof(double));

        for(int j=0; j < new_batch_size; j++) t->data[j] = learned_val;

    }
}

LinearLayer* _linear_layer(GraphContext* ctx, Model* m, int size){

    LinearLayer* llayer = (LinearLayer*)malloc(sizeof(LinearLayer));
    
    llayer->weights = RandParam(ctx, size);
    llayer->bias = RandParam(ctx, size);
    
    register_param(m, llayer->weights);
    register_param(m, llayer->bias);

    return llayer;
}

Tensor* Linear(GraphContext* ctx, LinearLayer* layer, Tensor* x){
    
    Tensor* wx = Mul(ctx, layer->weights, x);
    Tensor* y_hat = Add(ctx, wx, layer->bias);

    return y_hat;
}

DenseLayer* _dense_layer(GraphContext* ctx, Model* m, int input_dim, int output_dim, int batch_size){
    DenseLayer* dlayer = (DenseLayer*)malloc(sizeof(DenseLayer));

    dlayer->i_counts = input_dim;
    dlayer->o_counts = output_dim;

    dlayer->weights = (Tensor***)malloc(sizeof(Tensor**) * dlayer->o_counts);
    dlayer->bias = (Tensor**)malloc(sizeof(Tensor*) * dlayer->o_counts);

    for(int o=0; o<output_dim; o++){

        dlayer->weights[o] = (Tensor**)malloc(sizeof(Tensor*) * dlayer->i_counts);
        dlayer->bias[o] = RandParam(ctx, batch_size);
        register_param(m, dlayer->bias[o]);
        
        for(int i=0; i<input_dim; i++){

            dlayer->weights[o][i] = RandParam(ctx, batch_size);
            register_param(m, dlayer->weights[o][i]);

            double sync_val = dlayer->weights[o][i]->data[0];
            for(int k=0; k < batch_size; k++){
                dlayer->weights[o][i]->data[k] = sync_val;
            }
            
        }
    }

    return dlayer;
}



Tensor** Dense(GraphContext* ctx, DenseLayer* dlayer, Tensor** inputs){

    Tensor** output = (Tensor**)malloc(sizeof(Tensor*) * dlayer->o_counts);

    for(int o=0; o<dlayer->o_counts; o++){
        Tensor* sum = NULL;

        for(int i=0; i<dlayer->i_counts; i++){
            
            if(inputs[i]->size != dlayer->weights[o][i]->size){
                printf("DEBUG: Forward Mismatch! Input[%d] Size: %d | Weight[%d][%d] Size: %d\n",
                    i, inputs[i]->size, o, i, dlayer->weights[o][i]->size);
            }

            Tensor* wx = Mul(ctx, inputs[i], dlayer->weights[o][i]);
            
            if(sum == NULL) sum = wx;
            else sum = Add(ctx, sum, wx); 
        }

        output[o] = Add(ctx, sum, dlayer->bias[o]);
    }
    return output;
}

