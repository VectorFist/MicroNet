#include <omp.h>
#include "concatenate.h"

namespace micronet {

Concatenate::Concatenate(const int axis, const string& layer_name):
    Layer(layer_name, "Concatenate") {
    int_hps_["axis"] = axis;
}

chunk_ptr Concatenate::operator()(const vector<chunk_ptr>& in_chunks) {
    if (in_chunks.size() <= 1) {
        cout << "The number of concatenated chunks cannot be less 2!";
        exit(1);
    }
    int axis = int_hps_["axis"];
    if (axis != 0 && axis != 1 && axis != 2 && axis != 3) {
        cout << "Axis must be one of 0, 1, 2, 3!";
        exit(1);
    }

    string str_shape = in_chunks[0]->str_shape_exclude(axis);
    for (const auto& chunk: in_chunks) {
        if (chunk->str_shape_exclude(axis) != str_shape) {
            cout << "All chunks must have shape can be matched!";
            exit(1);
        }
    }

    chunks_in_ = in_chunks;
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Concatenate>(*this);
    for (const auto& chunk: in_chunks) {
        chunk->in_layers_.push_back(layer);
    }
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Concatenate::forward(bool is_train) {
    int axis = int_hps_["axis"];
    int num_out = 0, channels_out = 0, height_out = 0, width_out = 0;

    vector<int> out_shape = shape_inference();
    chunk_ptr chunk_out = chunks_out_[0];
    chunk_out->reshape(out_shape);

    float* out_data = chunk_out->data();
    for (const auto& chunk: chunks_in_) {
        const float* in_data = chunk->const_data();
        switch(axis) {
            case 0:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    const int n_out = num_out + n;
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n_out, c, h, w);
                                out_data[oindex] = in_data[iindex];
                            }
                        }
                    }
                }
                num_out += chunk->num();
                break;
            case 1:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        const int c_out = channels_out + c;
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c_out, h, w);
                                out_data[oindex] = in_data[iindex];
                            }
                        }
                    }
                }
                channels_out += chunk->channels();
                break;
            case 2:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            const int h_out = height_out + h;
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c, h_out, w);
                                out_data[oindex] = in_data[iindex];
                            }
                        }
                    }
                }
                height_out += chunk->height();
                break;
            case 3:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int w_out = width_out + w;
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c, h, w_out);
                                out_data[oindex] = in_data[iindex];
                            }
                        }
                    }
                }
                width_out += chunk->width();
        }
    }
    gradient_reset();
}

void Concatenate::backward() {
    int axis = int_hps_["axis"];
    int num_out = 0, channels_out = 0, height_out = 0, width_out = 0;

    chunk_ptr chunk_out = chunks_out_[0];

    const float* out_diff = chunk_out->const_diff();
    for (const auto& chunk: chunks_in_) {
        float* in_diff = chunk->diff();
        switch(axis) {
            case 0:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    const int n_out = num_out + n;
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n_out, c, h, w);
                                in_diff[iindex] += out_diff[oindex];
                            }
                        }
                    }
                }
                num_out += chunk->num();
                break;
            case 1:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        const int c_out = channels_out + c;
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c_out, h, w);
                                in_diff[iindex] += out_diff[oindex];
                            }
                        }
                    }
                }
                channels_out += chunk->channels();
                break;
            case 2:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            const int h_out = height_out + h;
                            for (int w = 0; w < chunk->width(); ++w) {
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c, h_out, w);
                                in_diff[iindex] += out_diff[oindex];
                            }
                        }
                    }
                }
                height_out += chunk->height();
                break;
            case 3:
                #pragma omp parallel for
                for (int n = 0; n < chunk->num(); ++n) {
                    for (int c = 0; c < chunk->channels(); ++c) {
                        for (int h = 0; h < chunk->height(); ++h) {
                            for (int w = 0; w < chunk->width(); ++w) {
                                int w_out = width_out + w;
                                const int iindex = chunk->offset(n, c, h, w);
                                const int oindex = chunk_out->offset(n, c, h, w_out);
                                in_diff[iindex] += out_diff[oindex];
                            }
                        }
                    }
                }
                width_out += chunk->width();
        }
    }
}

vector<int> Concatenate::shape_inference() {
    int axis = int_hps_["axis"];
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();

    int num_sum = 0, channels_sum = 0, height_sum = 0, width_sum = 0;
    for (const auto& chunk: chunks_in_)  {
        num_sum += chunk->num();
        channels_sum += chunk->channels();
        height_sum += chunk->height();
        width_sum += chunk->width();
    }

    switch(axis) {
        case 0: return {num_sum, channels, height, width};
        case 1: return {num, channels_sum, height, width};
        case 2: return {num, channels, height_sum, width};
        case 3: return {num, channels, height, width_sum};
    }
}

} // namespace micronet
