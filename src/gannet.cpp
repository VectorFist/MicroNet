
#include <queue>
#include <set>
#include "gannet.h"


namespace micronet {

GanNet::GanNet(const string& net_name): Net(net_name) {
}

GanNet::GanNet(chunk_ptr (*generator_constructor)(const chunk_ptr&), int noise_dim,
               chunk_ptr (*discriminator_constructor)(const chunk_ptr&), const vector<int>& real_shape,
               const string& net_name): Net(net_name) {
    auto noise = make_shared<Chunk>(1, noise_dim, 1, 1);
    vector<int> shape_tmp;
    if (real_shape.size() == 1) {
        shape_tmp = {1, real_shape[0], 1, 1};
    } else {
        shape_tmp = {1, real_shape[2], real_shape[0], real_shape[1]};
    }
    auto real = make_shared<Chunk>(shape_tmp);
    auto noise_label = make_shared<Chunk>(1, 1, 1, 1);
    auto real_label = make_shared<Chunk>(1, 1, 1, 1);
    auto generator_output = (*generator_constructor)(noise);
    add_layer_prefix(noise, generator_output, "generator");
    if (real->str_shape() != generator_output->str_shape()) {
        cout << "generator output shape must be same with real shape..." << endl;
        exit(1);
    }

    auto concat = Concatenate(0, "discriminator_concat")({real, generator_output});
    auto label_concat = Concatenate(0, "discriminator_concat")({real_label, noise_label});
    auto discriminator_output = (*discriminator_constructor)(concat);
    add_layer_prefix(concat, discriminator_output, "discriminator");

    auto discriminator_loss_prob = SigmoidLoss("discriminator_softmax_loss")(discriminator_output, label_concat);
    auto discriminator_loss = discriminator_loss_prob[0];
    auto generator_loss_prob = SigmoidLoss("generator_softmax_loss")(discriminator_output, label_concat);
    auto generator_loss = generator_loss_prob[0];

    key_chunks_["noise"] = noise;
    key_chunks_["real"] = real;
    key_chunks_["noise_label"] = noise_label;
    key_chunks_["real_label"] = real_label;
    key_chunks_["generator_loss"] = generator_loss;
    key_chunks_["discriminator_loss"] = discriminator_loss;
    key_chunks_["generator_output"] = generator_output;

    inputs_ = {real, real_label, noise, noise_label};

    initialize();
}

void GanNet::fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                      int batch_size, int epochs, int verbose, bool shuffle) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (!optimizer_) {
        cout << "optimizer must be assigned!" << endl;
        exit(1);
    }
    if (train_data.find("real") == train_data.end()) {
        cout << "train real data must be specified!" << endl;
        exit(1);
    }
    auto noise = key_chunks_["noise"];
    auto real = key_chunks_["real"];
    auto noise_label = key_chunks_["noise_label"];
    auto real_label = key_chunks_["real_label"];
    auto generator_loss = key_chunks_["generator_loss"];
    auto discriminator_loss = key_chunks_["discriminator_loss"];
    vector<int> noise_shape = noise->shape();
    vector<int> real_shape = real->shape();

    vector<data_t> real_data_vec {train_data.at("real")};
    DataProvider real_data(real_data_vec, true);
    int num_examples = real_data.num_samples();
    int num_fit_iters = num_examples * epochs / batch_size;
    int steps_per_epoch = num_examples / batch_size;

    optimizer_->total_iters_ = num_fit_iters;

    Timer step_timer;
    for (int epo = 0; epo < epochs; ++epo) {
        cout << "==================== epoch: " << epo+1 << " starts =================" << endl;
        for (int iter = 0; iter < steps_per_epoch; ++iter) {
            step_timer.resume();

            // optimizer discriminator
            noise->reshape(batch_size, noise_shape[1], noise_shape[2], noise_shape[3]);
            noise_label->reshape(batch_size, 1, 1, 1);
            real_label->reshape(batch_size, 1, 1, 1);
            //uniform_random_init(noise->count(), noise->data(), -1.0f, 1.0f);
            normal_random_init(noise->count(), noise->data(), 0.0f, 0.1f);
            //constant_init(noise_label->count(), noise_label->data(), 0.0f);
            uniform_random_init(noise_label->count(), noise_label->data(), 0.0f, 0.2f);
            //constant_init(real_label->count(), real_label->data(), 1.0f);
            uniform_random_init(real_label->count(), real_label->data(), 0.8f, 1.0f);
            if (iter_ % 500 == 0) {
                save_generator_imgs(iter_);
            }
            real_data.load_batch({real}, batch_size);
            iter_++;
            forward(true, "discriminator");
            backward("discriminator");
            update("discriminator");

            // optimizer generator
            real->reshape(0, real_shape[1], real_shape[2], real_shape[3]);
            real_label->reshape(0, 1, 1, 1);
            //constant_init(noise_label->count(), noise_label->data(), 1.0f);
            uniform_random_init(noise_label->count(), noise_label->data(), 0.8f, 1.0f);
            forward(true, "generator");
            backward("generator");
            update("generator");

            double time_used = step_timer.elapsed()*1000;
            if (iter % verbose == 0) {
                cout << "iter: " << setw(5) << iter << "/" << steps_per_epoch
                     << "\tgenerator loss: " << fixed <<setprecision(5) << generator_loss->const_data()[0]
                     << "\tdiscriminator loss: " << fixed <<setprecision(5) << discriminator_loss->const_data()[0]
                     << "\ttime_per_iter: " << fixed <<setprecision(5) << time_used << "ms" << endl;
            }
        }
    }
}

void GanNet::evaluate(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
}

data_t GanNet::inference(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (data.find("noise") == data.end()) {
        cout << "noise data must be specified!" << endl;
        exit(1);
    }

    Timer timer;
    vector<data_t> data_vec {data.at("noise")};
    DataProvider infer(data_vec, false);
    data_t data_inference;

    int infer_steps = infer.num_samples() / batch_size;
    int size_remain = infer.num_samples() % batch_size;
    for (int step = 0; step < infer_steps; ++step) {
        infer.load_batch({key_chunks_["noise"]}, batch_size);
        forward_generator(false);
        const float* generator_output_data = key_chunks_["generator_output"]->const_data();
        int dim = key_chunks_["generator_output"]->count() / batch_size;
        for (int i = 0; i < batch_size; ++i) {
            data_inference.push_back(vector<float>(generator_output_data, generator_output_data+dim));
            generator_output_data += dim;
        }
    }
    if (size_remain) {
        infer.load_batch({key_chunks_["noise"]}, size_remain);
        forward_generator(false);
        const float* generator_output_data = key_chunks_["generator_output"]->const_data();
        int dim = key_chunks_["generator_output"]->count() / size_remain;
        for (int i = 0; i < size_remain; ++i) {
            data_inference.push_back(vector<float>(generator_output_data, generator_output_data+dim));
            generator_output_data += dim;
        }
    }
    cout << "inference time used: " << timer.elapsed() << " s" << endl;

    return data_inference;
}

void GanNet::forward(bool is_train, const string& layer_prefix) {
    if (layer_prefix == "discriminator") {
        forward_generator(is_train);
        forward_discriminator(is_train);
        for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
            if ((*layer)->layer_name_ == "discriminator_softmax_loss")
                (*layer)->forward(is_train);
        }
    } else if (layer_prefix == "generator") {
        forward_discriminator(is_train);
        for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
            if ((*layer)->layer_name_ == "generator_softmax_loss")
                (*layer)->forward(is_train);
        }
    }
    //Timer t1;
    /*for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (layer_prefix == "discriminator" && (*layer)->layer_name_ == "generator_softmax_loss") {
            continue;
        }
        if (layer_prefix == "generator" && (*layer)->layer_name_ == "discriminator_softmax_loss") {
            continue;
        }
        Timer timer;
        (*layer)->forward();
        layer_op_time_[(*layer)->layer_name_].first = timer.elapsed()*1000;
    }*/
    //if (iter_ % 100 == 0) {
    //    cout << "forward: " << t1.elapsed()*1000 << endl;
    //}
}

inline bool starts_with(const string& str, const string& prefix) {
    return str.find(prefix) == 0;
}

void GanNet::backward(const string& layer_prefix) {
    //Timer t1;
    for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
        Timer timer;
        if (layer_prefix == "generator" && (*layer)->layer_name_ == "discriminator_softmax_loss") {
            continue;
        }
        if (layer_prefix == "discriminator" && starts_with((*layer)->layer_name_, "generator")) {
            continue;
        }
        if (layer_prefix == "discriminator" && (*layer)->layer_name_ == "discriminator_concat") {
            continue;
        }
        (*layer)->backward();
        layer_op_time_[(*layer)->layer_name_].second = timer.elapsed()*1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "backward: " << t1.elapsed()*1000 << endl;
    //}
}

void GanNet::update(const string& layer_prefix) {
    //Timer t1;
    for (auto& layer: net_sequences_) {
        Timer timer;
        if (layer_prefix == "generator" && starts_with(layer->layer_name_, "discriminator")) {
            continue;
        }
        if (layer_prefix == "discriminator" && starts_with(layer->layer_name_, "generator")) {
            continue;
        }
        for (auto& param: layer->params_) {
            if (param->trainable()) {
                optimizer_->optimize(param, iter_);
            }
        }
        layer_up_time_[layer->layer_name_] = timer.elapsed() * 1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "update: " << t1.elapsed()*1000 << endl;
    //}
}

void GanNet::forward_generator(bool is_train) {
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (starts_with((*layer)->layer_name_, "generator")
            && (*layer)->layer_name_ != "generator_softmax_loss") {
            (*layer)->forward(is_train);
        }
    }
}

void GanNet::forward_discriminator(bool is_train) {
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (starts_with((*layer)->layer_name_, "discriminator")
            && (*layer)->layer_name_ != "discriminator_softmax_loss") {
            (*layer)->forward(is_train);
        }
    }
}

void GanNet::save_generator_imgs(int iter) {
    string file = "gan_imgs/generator_iter_" + to_string(iter) + ".json";
    ofstream ofs(file);

    forward_generator(false);
    auto generator_output = key_chunks_["generator_output"];
    const float* data = generator_output->const_data();

    json j_imgs = {};
    int dim = generator_output->shape(1) * generator_output->shape(2) * generator_output->shape(3);
    for (int n = 0; n < generator_output->num(); ++n) {
        vector<float> img(data, data+dim);
        j_imgs.push_back(img);
        data += dim;
    }

    ofs << j_imgs << endl;
}

} // namespace micronet
