#include <string.h>

#include "deconvolution.h"
#include "util.h"
#include "math_func.h"

namespace micronet {

Deconvolution::Deconvolution(int kernel_h, int kernel_w, int stride_h, int stride_w,
                         int output_channels, const string& padding,
                         float mean, float stddev, float bias_value, const string& layer_name):
                         Layer(layer_name, "Deconvolution"), col_tmp_(new Chunk), all_one_tmp_(new Chunk) {
    if (padding != "same" && padding != "valid") {
        cout << "Padding must be same or valid !" << endl;
        exit(1);
    }
    str_hps_["padding"] = padding;
    int_hps_["kernel_h"] = kernel_h;
    int_hps_["kernel_w"] = kernel_w;
    int_hps_["stride_h"] = stride_h;
    int_hps_["stride_w"] = stride_w;
    int_hps_["output_channels"] = output_channels;
    flt_hps_["init_mean"] = mean;
    flt_hps_["init_stddev"] = stddev;
    flt_hps_["init_bias_value"] = bias_value;

    cout << "Initialize deconv layer: " << layer_name << " done..." << endl;
}

chunk_ptr Deconvolution::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    pad_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    params_.push_back(make_shared<Chunk>(in_chunk->channels(), int_hps_["output_channels"],
                                         int_hps_["kernel_h"], int_hps_["kernel_w"]));
    params_.push_back(make_shared<Chunk>(int_hps_["output_channels"], 1, 1, 1));
    initialize();

    layer_ptr layer = make_shared<Deconvolution>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Deconvolution::forward(bool is_train) {
    Timer timer_for;
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];
    int input_channels = chunks_in_[0]->channels();
    int output_channels = int_hps_["output_channels"];
    vector<int> out_shape = shape_inference();
    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int output_h = out_shape[2];
    int output_w = out_shape[3];
    int num = chunks_in_[0]->num();

    chunks_out_[0]->reshape(shape_inference());

    col_tmp_->reshape(output_channels*kernel_h*kernel_w, input_h*input_w, 1, 1);
    all_one_tmp_->reshape(output_h, output_w, 1, 1);
    all_one_tmp_->fill_value(1.0, 1.0);
    //output[0]->reshape(shape_inference(input[0]));

    const float* input_data = chunks_in_[0]->const_data();
    const float* weights_data = params_[0]->const_data();
    const float* bias_data = params_[1]->const_data();
    float* output_data = chunks_out_[0]->data();
    memset(output_data, 0, chunks_out_[0]->count()*sizeof(float));

    float img2col_time = 0, gemm_time = 0, bias_time = 0;
    for (int n = 0; n < num; ++n) {
        float* col_data = col_tmp_->data();
        const float* all_one_data = all_one_tmp_->const_data();

        //Timer timer;
        //img2col(input_data, input_channels, input_h, input_w, kernel_h, kernel_w,
        //        pad_h, pad_w, stride_h, stride_w, col_data);
        //img2col_time += timer.elapsed()*1000;
            //cout << "img2col time: " << timer.elapsed()*1000 << endl;

        //timer.resume();
        //gemm(0, 0, output_channels, output_h*output_w, input_channels*kernel_h*kernel_w, 1,
        //     weights_data, input_channels*kernel_h*kernel_w, col_data, output_h*output_w, 0,
        //     output_data, output_h*output_w);
        gemm(1, 0, output_channels*kernel_h*kernel_w, input_h*input_w, input_channels, 1,
             weights_data, output_channels*kernel_h*kernel_w, input_data, input_h*input_w, 0,
             col_data, input_h*input_w);
        col2img(col_data, output_channels, output_h, output_w, kernel_h, kernel_w, pad_h, pad_w,
                stride_h, stride_w, output_data);
        gemm(0, 0, output_channels, output_h*output_w, 1, 1,
             bias_data, 1, all_one_data, output_h*output_w, 1,
             output_data, output_h*output_w);
        //gemm_time += timer.elapsed()*1000;
            //cout << "gemm time: " << timer.elapsed()*1000 << endl;

        //timer.resume();
        /*float* output_data_tmp = output_data;
        for (int c = 0; c < output_channels; ++c) {
            add_scalar(output_h*output_w, bias_data[c], output_data_tmp);
            output_data_tmp += output_h * output_w;
        }*/
        //bias_time += timer.elapsed()*1000;
            //cout << "bias time: " << timer.elapsed()*1000 << endl;

        input_data += input_channels * input_h * input_w;
        output_data += output_channels * output_h * output_w;
    }
    //cout << "conv forward time:" << timer_for.elapsed()*1000 << endl;
    //exit(0);
    gradient_reset();
    //cout << "deconv forward" << endl;
}

void Deconvolution::backward() {
    Timer timer;
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];

    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int output_h = chunks_out_[0]->height();
    int output_w = chunks_out_[0]->width();
    int num = chunks_in_[0]->num();
    int input_channels = chunks_in_[0]->channels();
    int output_channels = chunks_out_[0]->channels();

    float* input_diff = chunks_in_[0]->diff();
    float* weights_diff = params_[0]->diff();
    float* bias_diff = params_[1]->diff();

    const float* input_data = chunks_in_[0]->const_data();
    const float* weights_data = params_[0]->const_data();
    const float* output_diff = chunks_out_[0]->const_diff();

    for (int n = 0; n < num; ++n) {
        float* col_diff = col_tmp_->diff();
        const float* all_one_data = all_one_tmp_->const_data();
        img2col(output_diff, output_channels, output_h, output_w, kernel_h, kernel_w,
                pad_h, pad_w, stride_h, stride_w, col_diff);

        gemm(0, 1, input_channels, output_channels*kernel_h*kernel_w, input_h*input_w, 1,
             input_data, input_h*input_w, col_diff, input_h*input_w, 1,
             weights_diff, output_channels*kernel_h*kernel_w);
        gemm(0, 0, input_channels, input_h*input_w, output_channels*kernel_h*kernel_w, 1,
             weights_data, output_channels*kernel_h*kernel_w, col_diff, input_h*input_w, 1,
             input_diff, input_h*input_w);

        gemm(0, 1, output_channels, 1, output_h*output_w, 1,
             output_diff, output_h*output_w, all_one_data, output_h*output_w, 1,
             bias_diff, 1);

        /*const float* output_diff_tmp = output_diff;
        for (int c = 0; c < output_channels; ++c) {
            bias_diff[c] = sum(output_h*output_w, bias_diff[c], output_diff_tmp);
            output_diff_tmp += output_h * output_w;
        }*/

        input_data += input_channels * input_h * input_w;
        input_diff += input_channels * input_h * input_w;
        output_diff += output_channels * output_h * output_w;
    }
}

void Deconvolution::initialize() {
    float* weights_data = params_[0]->data();
    float* bias_data = params_[1]->data();

    normal_random_init(params_[0]->count(), weights_data, flt_hps_["init_mean"], flt_hps_["init_stddev"]);
    constant_init(params_[1]->count(), bias_data, flt_hps_["init_bias_value"]);
}

void Deconvolution::pad_inference() {
    if (str_hps_["padding"] == "valid") {
        int_hps_["pad_h"] = 0;
        int_hps_["pad_w"] = 0;
    } else if (str_hps_["padding"] == "same") {
        int kernel_h = int_hps_["kernel_h"];
        int kernel_w = int_hps_["kernel_w"];
        int stride_h = int_hps_["stride_h"];
        int stride_w = int_hps_["stride_w"];

        int input_h = chunks_in_[0]->height();
        int input_w = chunks_in_[0]->width();
        int output_h = input_h * stride_h;
        int output_w = input_w * stride_w;
        int_hps_["pad_h"] = ((input_h - 1) * stride_h + kernel_h - output_h) / 2;
        int_hps_["pad_w"] = ((input_w - 1) * stride_w + kernel_w - output_w) / 2;
    }
}

vector<int> Deconvolution::shape_inference() {
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];

    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int output_h = (input_h - 1) * stride_h + kernel_h - 2*pad_h;
    int output_w = (input_w - 1) * stride_w + kernel_w - 2*pad_w;
    if (str_hps_["padding"] == "same") {
        output_h = input_h * stride_h;
        output_w = input_w * stride_w;
    }
    int num = chunks_in_[0]->num();
    int output_channels = int_hps_["output_channels"];

    return {num, output_channels, output_h, output_w};
}

} // namespace micronet
