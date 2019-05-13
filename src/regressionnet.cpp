#include "regressionnet.h"

namespace micronet {

void print_loss(int iter, int num_iters, float loss, double time_per_iter) {
    cout << "iter: " << setw(5) << iter << "/" << num_iters << "\tloss: " << fixed <<setprecision(5) <<
        loss << "\ttime_per_iter: " << fixed << setprecision(5) << (time_per_iter * 1000) << " ms" << endl;
}

RegressionNet::RegressionNet(const string& net_name): Net(net_name) {
}

RegressionNet::RegressionNet(chunk_ptr (*graph_constructor)(const vector<chunk_ptr>&),
                             const vector<vector<int>>& input_shapes, const string& net_name): Net(net_name) {
    vector<chunk_ptr> graph_inputs;
    for (int i = 0; i < input_shapes.size(); ++i) {
        auto input = make_shared<Chunk>(1, input_shapes[i][0], input_shapes[i][1], input_shapes[i][2]);
        graph_inputs.push_back(input);
        key_chunks_["input"+to_string(i)] = input;
    }

    auto output = (*graph_constructor)(graph_inputs);
    key_chunks_["output"] = output;
    key_chunks_["target"] = make_shared<Chunk>(output->shape());
    auto loss = L2Loss()(output, key_chunks_["target"]);
    key_chunks_["loss"] = loss;

    inputs_ = graph_inputs;
    inputs_.push_back(key_chunks_["target"]);
    initialize();
}

void RegressionNet::fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                      int batch_size, int epochs, int verbose, bool shuffle) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (!optimizer_) {
        cout << "optimizer must be assigned!" << endl;
        exit(1);
    }
    for (int i = 0; i < inputs_.size()-1; ++i) {
        if (train_data.find("input"+to_string(i)) == train_data.end()) {
            cout << "train input" << i << " data must be specified!" << endl;
            exit(1);
        }
        if (valid_data.find("input"+to_string(i)) == valid_data.end()) {
            cout << "valid input" << i << " data must be specified!" << endl;
            exit(1);
        }
    }
    if (train_data.find("target") == train_data.end()) {
        cout << "train target data must be specified!" << endl;
        exit(1);
    }
    if (valid_data.find("target") == valid_data.end()) {
        cout << "valid target data must be specified!" << endl;
        exit(1);
    }

    vector<data_t> train_data_vec, valid_data_vec;
    for (int i = 0; i < inputs_.size()-1; ++i) {
        train_data_vec.push_back(train_data.at("input"+to_string(i)));
        //valid_data_vec.push_back(valid_data.at("input"+to_string(i)));
    }
    train_data_vec.push_back(train_data.at("target"));
    //valid_data_vec.push_back(valid_data.at("target"));
    DataProvider train(train_data_vec, shuffle);
    //DataProvider valid(valid_data_vec, false);

    int train_num_examples = train.num_samples();
    int train_num_fit_iters = train_num_examples * epochs / batch_size;
    optimizer_->total_iters_ = train_num_fit_iters;

    int steps_per_epoch = train_num_examples / batch_size;
    for (int epo = 0; epo < epochs; ++epo) {
        cout << "==================== epoch: " << epo+1 << " starts =================" << endl;
        float train_loss = 0;
        Timer epoch_timer, step_timer;
        for (int step = 0; step < steps_per_epoch; ++step) {
            step_timer.resume();
            train.load_batch(inputs_, batch_size);
            forward(true); //cout << "forward " << step << endl;
            backward(); //cout << "backward " << step << endl;
            update(); //cout << "update " << step << endl;
            float loss = key_chunks_["loss"]->const_data()[0];
            train_loss += loss;
            if (step % verbose == 0) {
                print_loss(step, steps_per_epoch, loss, step_timer.elapsed());
            }
        }
        train_loss /= steps_per_epoch;
        cout << "epoch: " << epo+1 << ", train avg loss: " << train_loss <<
                ", time_per_train_epoch: " << fixed << setprecision(4) << epoch_timer.elapsed() << "s" << endl;

        /*for (const auto& layer: net_sequences_) {
            string layer_name = layer->layer_name_;
            cout << layer_name << "\t\t" << "forward time: " << layer_op_time_[layer_name].first << " ms\t\t" <<
                    "backward time: " << layer_op_time_[layer_name].second << " ms\t\t" <<
                    "update time: " << layer_up_time_[layer_name] << " ms" << endl;
        }*/

        cout << "epoch: " << epo+1 << ", ";
        evaluate(valid_data, 50);
        cout << endl;
    }
}

void RegressionNet::evaluate(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    for (int i = 0; i < inputs_.size()-1; ++i) {
        if (data.find("input"+to_string(i)) == data.end()) {
            cout << "input" << i << " data must be specified!" << endl;
            exit(1);
        }
    }
    if (data.find("target") == data.end()) {
        cout << "target data must be specified!" << endl;
        exit(1);
    }

    Timer timer;
    vector<data_t> eval_data_vec;
    for (int i = 0; i < inputs_.size()-1; ++i) {
        eval_data_vec.push_back(data.at("input"+to_string(i)));
    }
    eval_data_vec.push_back(data.at("target"));
    DataProvider eval(eval_data_vec, false);

    int eval_fit_steps = eval.num_samples() / batch_size;
    int size_remain = eval.num_samples() % batch_size;
    float eval_loss = 0;
    for (int step = 0; step < eval_fit_steps; ++step) {
        eval.load_batch(inputs_, batch_size);
        //cout << step << endl;
        forward(false);
        float loss = key_chunks_["loss"]->const_data()[0];
        eval_loss += loss * batch_size;
    }
    if (size_remain) {
        eval.load_batch(inputs_, size_remain);
        //cout << step << endl;
        forward(false);
        float loss = key_chunks_["loss"]->const_data()[0];
        eval_loss += loss * size_remain;
    }
    eval_loss /= eval.num_samples();
    cout << "eval loss: " << eval_loss << ", time used: " << fixed
         << setprecision(4) << timer.elapsed() << "s" << endl;
}

data_t RegressionNet::inference(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    for (int i = 0; i < inputs_.size()-1; ++i) {
        if (data.find("input"+to_string(i)) == data.end()) {
            cout << "input" << i << " data must be specified!" << endl;
            exit(1);
        }
    }

    Timer timer;
    vector<chunk_ptr> inputs;
    vector<data_t> data_vec;
    for (int i = 0; i < inputs_.size()-1; ++i) {
        data_vec.push_back(data.at("input"+to_string(i)));
        inputs.push_back(key_chunks_["input"+to_string(i)]);
    }
    DataProvider infer(data_vec, false);
    data_t data_inference;

    int infer_steps = infer.num_samples() / batch_size;
    int size_remain = infer.num_samples() % batch_size;
    //cout << key_chunks_["output"]->count() << key_chunks_["output"]->num() << endl;
    int dim = key_chunks_["output"]->count() / key_chunks_["output"]->num();
    vector<int> target_shape = {key_chunks_["output"]->shape(1), key_chunks_["output"]->shape(2), key_chunks_["output"]->shape(3)};
    for (int step = 0; step < infer_steps; ++step) {
        infer.load_batch(inputs, batch_size);
        key_chunks_["target"]->reshape(batch_size, target_shape[0], target_shape[1], target_shape[2]);
        forward(false);
        const float* output_data = key_chunks_["output"]->const_data();
        for (int i = 0; i < batch_size; ++i) {
            data_inference.push_back(vector<float>(output_data, output_data+dim));
            output_data += dim;
        }
    }
    if (size_remain) {
        infer.load_batch(inputs, size_remain);
        key_chunks_["target"]->reshape(size_remain, target_shape[0], target_shape[1], target_shape[2]);
        forward(false);
        const float* output_data = key_chunks_["output"]->const_data();
        for (int i = 0; i < batch_size; ++i) {
            data_inference.push_back(vector<float>(output_data, output_data+dim));
            output_data += dim;
        }
    }
    cout << "time used: " << timer.elapsed() << " s" << endl;

    return data_inference;
}

void RegressionNet::forward(bool is_train, const string& layer_prefix) {
    //Timer t1;
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        Timer timer;
        (*layer)->forward(is_train); //cout << (*layer)->layer_name_ << "forward" << endl;
        layer_op_time_[(*layer)->layer_name_].first = timer.elapsed()*1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "forward: " << t1.elapsed()*1000 << endl;
    //}
}

void RegressionNet::backward(const string& layer_prefix) {
    //Timer t1;
    for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
        Timer timer;
        (*layer)->backward();
        layer_op_time_[(*layer)->layer_name_].second = timer.elapsed()*1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "backward: " << t1.elapsed()*1000 << endl;
    //}
}

void RegressionNet::update(const string& layer_prefix) {
    //Timer t1;
    iter_++;
    for (auto& layer: net_sequences_) {
        Timer timer;
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

} // namespace micronet
