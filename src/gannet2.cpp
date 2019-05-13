
#include <queue>
#include <set>
#include "gannet2.h"
#include "concatenate.h"
#include "sigmoidloss.h"
#include "add.h"


namespace micronet {

GanNet2::GanNet2(const string& net_name): Net(net_name) {
}

GanNet2::GanNet2(chunk_ptr (*generator_constructor)(const chunk_ptr&), int noise_dim,
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

    auto discriminator_real_output = (*discriminator_constructor)(real);
    add_layer_prefix(real, discriminator_real_output, "discriminator_real");
    share_parameters(real, discriminator_real_output, "discriminator");
    auto discriminator_fake_output = (*discriminator_constructor)(generator_output);
    add_layer_prefix(generator_output, discriminator_fake_output, "discriminator_fake");
    share_parameters(generator_output, discriminator_fake_output, "discriminator");

    auto discriminator_real_loss_prob = SigmoidLoss("discriminator_loss_real")(discriminator_real_output, real_label);
    auto discriminator_real_loss = discriminator_real_loss_prob[0];
    auto discriminator_fake_loss_prob = SigmoidLoss("discriminator_loss_fake")(discriminator_fake_output, noise_label);
    auto discriminator_fake_loss = discriminator_fake_loss_prob[0];
    auto discriminator_loss = Add("discriminator_loss")(discriminator_real_loss, discriminator_fake_loss);

    auto generator_loss_prob = SigmoidLoss("generator_loss")(discriminator_fake_output, noise_label);
    auto generator_loss = generator_loss_prob[0];

    key_chunks_["noise"] = noise;
    key_chunks_["real"] = real;
    key_chunks_["noise_label"] = noise_label;
    key_chunks_["real_label"] = real_label;
    key_chunks_["generator_loss"] = generator_loss;
    key_chunks_["discriminator_real_loss"] = discriminator_real_loss;
    key_chunks_["discriminator_fake_loss"] = discriminator_fake_loss;
    key_chunks_["discriminator_loss"] = discriminator_loss;
    key_chunks_["generator_output"] = generator_output;

    inputs_ = {real, real_label, noise, noise_label};

    initialize();
    //exit(0);
}

void GanNet2::fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
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
    auto discriminator_real_loss = key_chunks_["discriminator_real_loss"];
    auto discriminator_fake_loss = key_chunks_["discriminator_fake_loss"];
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
            normal_random_init(noise->count(), noise->data(), 0.0f, 0.1f);
            uniform_random_init(noise_label->count(), noise_label->data(), 0.0f, 0.2f);
            real_data.load_batch({real}, batch_size);
            uniform_random_init(real_label->count(), real_label->data(), 0.8f, 1.0f);
            if (iter_ % 500 == 0) {
                save_generator_imgs(iter_);
            }
            iter_++;
            forward(true, "discriminator");
            backward("discriminator");
            update("discriminator");

            // optimizer generator
            uniform_random_init(noise_label->count(), noise_label->data(), 0.8f, 1.0f);
            forward(true, "generator");
            backward("generator");
            update("generator");

            double time_used = step_timer.elapsed()*1000;
            if (iter % verbose == 0) {
                cout << "iter: " << setw(4) << iter << "/" << steps_per_epoch
                     << "\tgenerator loss: " << fixed <<setprecision(4) << generator_loss->const_data()[0]
                     << "\tdiscriminator loss: " << fixed <<setprecision(4) << discriminator_loss->const_data()[0]
                     << "(real_loss: " << fixed <<setprecision(4) << discriminator_real_loss->const_data()[0]
                     << ", fake_loss: " << fixed <<setprecision(4) << discriminator_fake_loss->const_data()[0] << ")"
                     << "\ttime_per_iter: " << fixed <<setprecision(4) << time_used << "ms" << endl;
            }
        }
    }
}

void GanNet2::evaluate(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
}

data_t GanNet2::inference(const map<string, data_t>& data, int batch_size) {
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

void GanNet2::forward(bool is_train, const string& layer_prefix) {
    if (layer_prefix == "discriminator") {
        forward_generator(is_train);
        forward_discriminator_real(is_train);
        forward_discriminator_fake(is_train);
        for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
            if ((*layer)->layer_name_ == "discriminator_loss_real" ||
                (*layer)->layer_name_ == "discriminator_loss_fake" ||
                (*layer)->layer_name_ == "discriminator_loss")
                (*layer)->forward(is_train);
        }
    } else if (layer_prefix == "generator") {
        forward_discriminator_fake(is_train);
        for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
            if ((*layer)->layer_name_ == "generator_loss")
                (*layer)->forward(is_train);
        }
    }

    //if (iter_ % 100 == 0) {
    //    cout << "forward: " << t1.elapsed()*1000 << endl;
    //}
}

inline bool starts_with(const string& str, const string& prefix) {
    return str.find(prefix) == 0;
}

void GanNet2::backward(const string& layer_prefix) {
    //Timer t1;
    if (layer_prefix == "discriminator") {
        float* discriminator_loss_diff = key_chunks_["discriminator_loss"]->diff();
        discriminator_loss_diff[0] = 1;
        for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
            if (starts_with((*layer)->layer_name_, "discriminator"))
                (*layer)->backward();
        }
    } else if (layer_prefix == "generator") {
        float* generator_loss_diff = key_chunks_["generator_loss"]->diff();
        generator_loss_diff[0] = 1;
        auto generator_output = key_chunks_["generator_output"];
        float* generator_output_diff = generator_output->diff();
        constant_init(generator_output->count(), generator_output_diff, 0);
        for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
            if (starts_with((*layer)->layer_name_, "generator") ||
                starts_with((*layer)->layer_name_, "discriminator_fake"))
                (*layer)->backward();
        }
    }
    //if (iter_ % 100 == 0) {
    //    cout << "backward: " << t1.elapsed()*1000 << endl;
    //}
}

void GanNet2::update(const string& layer_prefix) {
    //Timer t1;
    if (layer_prefix == "discriminator") {
        for (auto& layer: net_sequences_) {
            if (starts_with(layer->layer_name_, "discriminator_real")) {
                for (auto& param: layer->params_) {
                    if (param->trainable()) {
                        optimizer_->optimize(param, iter_);
                    }
                }
            }
        }
    } else if (layer_prefix == "generator") {
        for (auto& layer: net_sequences_) {
            if (starts_with(layer->layer_name_, "generator")) {
                for (auto& param: layer->params_) {
                    if (param->trainable()) {
                        optimizer_->optimize(param, iter_);
                    }
                }
            }
        }
    }

    //if (iter_ % 100 == 0) {
    //    cout << "update: " << t1.elapsed()*1000 << endl;
    //}
}

void GanNet2::forward_generator(bool is_train) {
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (starts_with((*layer)->layer_name_, "generator")
            && (*layer)->layer_name_ != "generator_loss") {
            (*layer)->forward(is_train);
        }
    }
}

void GanNet2::forward_discriminator_real(bool is_train) {
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (starts_with((*layer)->layer_name_, "discriminator_real")) {
            (*layer)->forward(is_train);
        }
    }
}

void GanNet2::forward_discriminator_fake(bool is_train) {
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        if (starts_with((*layer)->layer_name_, "discriminator_fake")) {
            (*layer)->forward(is_train);
        }
    }
}

void GanNet2::save_generator_imgs(int iter) {
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
