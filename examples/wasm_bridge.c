#include <emscripten.h>
#include <stdlib.h>
#include <stdio.h>

#include "graph.h"
#include "layers.h"
#include "optimizers.h"
#include "ops.h"

// Compile:
//   emcc wasm_bridge.c src/*.c -Iinclude -lm \
//     -s EXPORTED_FUNCTIONS="['_init_network','_train_epoch','_predict_grid','_malloc','_free']" \
//     -s EXPORTED_RUNTIME_METHODS="['cwrap','HEAPF64']" \
//     -s ALLOW_MEMORY_GROWTH=1 \
//     -o net.js

static GraphContext* ctx = NULL;
static Model*        model = NULL;
static DenseLayer*   d1 = NULL;
static DenseLayer*   d2 = NULL;
static Optimizer*    opt = NULL;
static int           current_batch_size = 4;
static int           current_hidden = 4;

// (Re)initialize the network. Safe to call multiple times (previous state is freed).
// batch_size = number of training samples per epoch (full-batch for small datasets).
EMSCRIPTEN_KEEPALIVE
void init_network(int hidden_neurons, double learning_rate, int batch_size) {

    if (ctx) {
        reset_graph(ctx);
        free(ctx->tape);
        free(ctx->params);
        free(ctx->inputs);
        free(ctx);
    }
    if (model) { free(model->params); free(model); }
    if (opt)   { free(opt->velocity); free(opt); }

    current_batch_size = batch_size;
    current_hidden     = hidden_neurons;

    ctx   = init_graph(10000, 1000, 256);
    model = init_model(256);

    d1 = _dense_layer(ctx, model, 2,              hidden_neurons, batch_size);
    d2 = _dense_layer(ctx, model, hidden_neurons, 1,              batch_size);

    opt = sgd(model->params, model->count, batch_size, learning_rate, 0.9);
}

// Train one full epoch. Returns MSE loss averaged over the batch.
EMSCRIPTEN_KEEPALIVE
double train_epoch(double* x0_data, double* x1_data, double* y_data) {

    Tensor* inputs[2];
    inputs[0] = Input(ctx, x0_data, current_batch_size);
    inputs[1] = Input(ctx, x1_data, current_batch_size);
    Tensor* target = Input(ctx, y_data, current_batch_size);

    Tensor** h1     = Dense(ctx, d1, inputs);
    Tensor** h1_act = (Tensor**)malloc(sizeof(Tensor*) * current_hidden);
    for (int i = 0; i < current_hidden; i++) h1_act[i] = Tanh(ctx, h1[i]);

    Tensor** out  = Dense(ctx, d2, h1_act);
    Tensor*  pred = Sigmoid(ctx, out[0]);
    Tensor*  loss = Mse(ctx, target, pred);

    backward(ctx, loss);
    opt->step(opt); // internally: sync_weights + momentum update + zero_grad

    double current_loss = loss->data[0];

    reset_tape(ctx);
    free(h1_act);
    free(h1);
    free(out);

    return current_loss;
}

// Run forward pass over an arbitrary grid/point set.
// Temporarily resizes model weights to num_points, infers, then restores batch_size.
EMSCRIPTEN_KEEPALIVE
void predict_grid(double* grid_x0, double* grid_x1, double* out_preds, int num_points) {

    resize_model_batch(model, num_points);

    Tensor* inputs[2];
    inputs[0] = Input(ctx, grid_x0, num_points);
    inputs[1] = Input(ctx, grid_x1, num_points); // was grid_x0 — fixed

    Tensor** h1     = Dense(ctx, d1, inputs);
    Tensor** h1_act = (Tensor**)malloc(sizeof(Tensor*) * current_hidden);
    for (int i = 0; i < current_hidden; i++) h1_act[i] = Tanh(ctx, h1[i]);

    Tensor** out  = Dense(ctx, d2, h1_act);
    Tensor*  pred = Sigmoid(ctx, out[0]);

    for (int i = 0; i < num_points; i++) out_preds[i] = pred->data[i];

    reset_tape(ctx);
    free(h1_act);
    free(h1);
    free(out);

    resize_model_batch(model, current_batch_size);
}
