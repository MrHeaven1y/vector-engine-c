#ifndef IO_H
#define IO_H

#include<stdio.h>
#include "stensor.h"
#include "layers.h"

#define MODEL_MAGIC 0xFE12BA

void save_model(Tensor* root,int max_nodes, const char* filename);
int load_model(const char* filename);

void export_model_header(Tensor** params, int count, const char* filename);

#endif