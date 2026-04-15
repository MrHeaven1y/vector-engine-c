#ifndef TYPES_H
#define TYPES_H

typedef enum {
    OP_VAR,
    OP_NEG,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_SQUARE,
    OP_DIV,
    OP_EXP,
    OP_LOG,
    OP_MEAN,
    OP_RELU,
    OP_TANH,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_MSE,
    OP_RMSE,
    OP_MAE,
    OP_SOFTMAX_CE
} OpType;

#endif