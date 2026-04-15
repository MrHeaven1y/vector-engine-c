#ifndef OPS_H
#define OPS_H

#include "stensor.h"
#include "graph.h"


// Forward Operations (vector form)

Tensor* Add(GraphContext* ctx, Tensor* a, Tensor* b); 
Tensor* Mul(GraphContext* ctx, Tensor* a, Tensor* b); 
Tensor* Sub(GraphContext* ctx, Tensor* a, Tensor* b); 
Tensor* Div(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* Neg(GraphContext* ctx, Tensor* a);
Tensor* Square(GraphContext* ctx, Tensor* a); 
Tensor* Exp(GraphContext* ctx, Tensor* a);
Tensor* Log(GraphContext* ctx, Tensor* a);
Tensor* Mean(GraphContext* ctx, Tensor* a);

// backward operations (Reverse mode autodiff)
void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_sub(Tensor* t);
void backward_square(Tensor* t);
void backward_div(Tensor* t);
void backward_neg(Tensor* t);
void backward_exp(Tensor* t);
void backward_log(Tensor* t);
void backward_mean(Tensor* t);

void backward_relu(Tensor* a);
void backward_tanh(Tensor* a);
void backward_sigmoid(Tensor* a);
void backward_softmax(Tensor* a);

void backward_mse(Tensor* a);
void backward_rmse(Tensor* a);
void backward_mae(Tensor* a);
void backward_softmax_CE(Tensor* a);
void backward_binary_crossentropy(Tensor* a);
void backward_CE(Tensor* a);
void backward_binary_crossentropy_sigmoid(Tensor* a);

//Activations
Tensor* Relu(GraphContext* ctx, Tensor* a);
Tensor* Tanh(GraphContext* ctx, Tensor* a);
Tensor* Sigmoid(GraphContext* ctx, Tensor* a);
Tensor* Softmax(GraphContext* ctx, Tensor* a, int sample_width);

//Loss Functions

Tensor* Mse(GraphContext* ctx,Tensor* y, Tensor* y_hat);
Tensor* Rmse(GraphContext* ctx,Tensor* y, Tensor* y_hat);
Tensor* Mae(GraphContext* ctx,Tensor* y, Tensor* y_hat);

// fused loss funcs
Tensor* Softmax_CE(GraphContext* ctx, Tensor* logits, Tensor* target);
Tensor* Binary_CE(GraphContext* ctx, Tensor* logits, Tensor* target);
#endif