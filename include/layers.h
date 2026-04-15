#ifndef LAYERS_H
#define LAYERS_H

#include "stensor.h"
#include "graph.h"

typedef struct Model{
    
    int count;
    int cap;
    Tensor** params;
    
} Model;

typedef struct DenseLayer{

    Tensor*** weights;
    Tensor** bias;
    int i_counts; // input counts
    int o_counts; // output counts

} DenseLayer;


typedef struct LinearLayer{

    Tensor* weights;
    Tensor* bias;

} LinearLayer;

void resize_model_batch(Model* m, int new_batch_size);

Model* init_model(int capacity);
void register_param(Model* m, Tensor* param);

LinearLayer* _linear_layer(GraphContext* ctx, Model* m, int size);
Tensor* Linear(GraphContext* ctx, LinearLayer* llayer, Tensor* x);

DenseLayer* _dense_layer(GraphContext* ctx, Model* m, int input_dim, int output_dim, int batch_size);
Tensor** Dense(GraphContext* ctx, DenseLayer* dlayer, Tensor** inputs);

#endif