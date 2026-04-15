#ifndef GRAPH_H
#define GRAPH_H

#include "stensor.h"

typedef struct GraphContext
{
    Tensor **tape;
    Tensor **params;
    Tensor **inputs;

    int t_cap, t_count;
    int i_cap, i_count;
    int p_cap, p_count;

} GraphContext;

GraphContext *init_graph(int t_cap, int p_cap, int i_cap);
Tensor *_alloc_node(GraphContext *ctx,
                    OpType type,
                    int size,
                    Tensor *left,
                    Tensor *right,
                    int is_param); // we'll later add view and storage concept here

void _track_ops(GraphContext *ctx, Tensor *t);
void _track_inputs(GraphContext *ctx, Tensor *t);
void _track_params(GraphContext *ctx, Tensor *t);

Tensor *Param(GraphContext *ctx, double *values, int size);
Tensor *RandParam(GraphContext *ctx, int size);
Tensor *Input(GraphContext *ctx, double *values, int size);

void free_tensor(Tensor *t);
void reset_tape(GraphContext *ctx);
void reset_inputs(GraphContext *ctx);
void reset_params(GraphContext *ctx);
void reset_graph(GraphContext *ctx);
void reset_computation(GraphContext *ctx);

Tensor **build_softmax(GraphContext *ctx, Tensor **logits, int classes);
Tensor *build_categorical_CE(GraphContext *ctx, Tensor **probs, Tensor **targets, int num_classes);


#endif