#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "stensor.h"

typedef struct Optimizer Optimizer;

typedef void (*step) (struct Optimizer* opt);

typedef struct Optimizer{
    
    Tensor** params;
    double** velocity;
    int p_count;

    double lr;
    double beta;
    int batch_size;

    step step;


} Optimizer;

void zero_grad(Optimizer* optim);
void clip_grad_norm(Optimizer* optim, double max_norm);
void sync_weights(Optimizer* optim);
void _sgd_momentum(Optimizer* optim);
Optimizer* sgd(Tensor** params, int count, int batch_size, double lr, double beta);

#endif