#ifndef TENSOR_H
#define TENSOR_H


#include "types.h"
#include<stdio.h>
#include<stdlib.h>

typedef struct GraphContext GraphContext;

typedef struct Tensor Tensor;

typedef void (*BackwardFunc) (struct Tensor* self);

struct Tensor{
    
    double* data;
    double* grad;

    int size;

    Tensor* left;
    Tensor* right;

    int visited;
    int aux_param;

    OpType type;

    BackwardFunc op_backward;
};

void _build_topo(Tensor* v, Tensor*** sorted, int* idx, int* cap);
void backward(GraphContext* ctx, Tensor* root);
void reset_visited(Tensor* root);

void shape(Tensor* v);
void print_tensor(Tensor* v);

#endif