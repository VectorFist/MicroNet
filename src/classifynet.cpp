#include "classifynet.h"


namespace micronet {

void print_loss_acc(int iter, int num_iters, float loss, double acc, double time_per_iter) {
    cout << "iter: " << setw(5) << iter << "/" << num_iters << "\tloss: " << fixed <<setprecision(5) <<
        loss << "\tacc: " << fixed << setprecision(5) << acc  <<
        "\ttime_per_iter: " << fixed << setprecision(5) << (time_per_iter * 1000) << " ms" << endl;
}

ClassifyNet::ClassifyNet(const string& net_name): Net(net_name) {
}

ClassifyNet::ClassifyNet(chunk_ptr (*graph_constructor)(const chunk_ptr&), const vector<int>& img_shape,
                         const string& net_name): Net(net_name) {
    auto img = make_shared<Chunk>(1, img_shape[2], img_shape[0], img_shape[1]);
    auto label = make_shared<Chunk>(1, 1, 1, 1);
    auto output = (*graph_constructor)(img);
    auto loss_prob = SoftmaxLoss()(output, label);
    auto acc = Accuracy()(loss_prob[1], label);
    auto argmax = ArgMax()(output);

    key_chunks_["img"] = img;
    key_chunks_["label"] = label;
    key_chunks_["loss"] = loss_prob[0];
    key_chunks_["acc"] = acc;
    key_chunks_["argmax"] = argmax;
    inputs_ = {img, label};

    initialize();
}

void ClassifyNet::fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                      int batch_size, int epochs, int verbose, bool shuffle) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (!optimizer_) {
        cout << "optimizer must be assigned!" << endl;
        exit(1);
    }
    if (train_data.find("img") == train_data.end()) {
        cout << "train img data must be specified!" << endl;
        exit(1);
    }
    if (train_data.find("label") == train_data.end()) {
        cout << "train label data must be specified!" << endl;
        exit(1);
    }
    if (valid_data.find("img") == valid_data.end()) {
        cout << "valid img data must be specified!" << endl;
        exit(1);
    }
    if (valid_data.find("label") == valid_data.end()) {
        cout << "valid label data must be specified!" << endl;
        exit(1);
    }
    vector<data_t> train_data_vec = {train_data.at("img"), train_data.at("label")};
    vector<data_t> valid_data_vec = {valid_data.at("img"), valid_data.at("label")};
    DataProvider train(train_data_vec, shuffle);
    DataProvider valid(valid_data_vec, false);  //cout << "shit\n";

    int train_num_examples = train.num_samples();
    int train_num_fit_iters = train_num_examples * epochs / batch_size;
    optimizer_->total_iters_ = train_num_fit_iters;

    int steps_per_epoch = train_num_examples / batch_size;
    for (int epo = 0; epo < epochs; ++epo) {
        cout << "==================== epoch: " << epo+1 << " starts =================" << endl;
        //iter_++;
        float train_loss = 0;
        float train_acc = 0;
        Timer epoch_timer, step_timer;
        for (int step = 0; step < steps_per_epoch; ++step) {
            step_timer.resume();
            train.load_batch(inputs_, batch_size);
            forward(true);
            backward();
            update();
            float loss = key_chunks_["loss"]->const_data()[0];
            float acc = key_chunks_["acc"]->const_data()[0];
            train_loss += loss;
            train_acc += acc;
            if (step % verbose == 0) {
                print_loss_acc(step, steps_per_epoch, loss, acc, step_timer.elapsed());
            }
        }
        train_loss /= steps_per_epoch;
        train_acc /= steps_per_epoch;
        cout << "epoch: " << epo+1 << ", train avg loss: " << train_loss << ", train avg acc: " << train_acc <<
                ", time_per_train_epoch: " << fixed << setprecision(4) << epoch_timer.elapsed() << "s" << endl;

        /*for (const auto& layer: net_sequences_) {
            string layer_name = layer->layer_name_;
            cout << layer_name << "\t\t" << "forward time: " << layer_op_time_[layer_name].first << " ms\t\t" <<
                    "backward time: " << layer_op_time_[layer_name].second << " ms\t\t" <<
                    "update time: " << layer_up_time_[layer_name] << " ms" << endl;
        }*/

        cout << "epoch: " << epo+1 << ", ";
        evaluate(valid_data, 100);
        cout << endl;
    }
}

void ClassifyNet::evaluate(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (data.find("img") == data.end()) {
        cout << "img data must be specified!" << endl;
        exit(1);
    }
    if (data.find("label") == data.end()) {
        cout << "label data must be specified!" << endl;
        exit(1);
    }

    Timer timer;
    vector<data_t> data_vec {data.at("img"), data.at("label")};
    DataProvider eval(data_vec, false);
    int eval_fit_steps = eval.num_samples() / batch_size;
    int size_remain = eval.num_samples() % batch_size;
    float eval_loss = 0;
    float eval_acc = 0;
    for (int step = 0; step < eval_fit_steps; ++step) {
        eval.load_batch(inputs_, batch_size);
        //cout << step << endl;
        forward(false);
        float loss = key_chunks_["loss"]->const_data()[0];
        float acc = key_chunks_["acc"]->const_data()[0];
        eval_loss += loss * batch_size;
        eval_acc += acc * batch_size;
    }
    if (size_remain) {
        eval.load_batch(inputs_, size_remain);
        //cout << step << endl;
        forward(false);
        float loss = key_chunks_["loss"]->const_data()[0];
        float acc = key_chunks_["acc"]->const_data()[0];
        eval_loss += loss * size_remain;
        eval_acc += acc * size_remain;
    }
    eval_loss /= eval.num_samples();
    eval_acc /= eval.num_samples();
    cout << "eval loss: " << eval_loss << ", eval acc: " << eval_acc <<
            ", time used: " << fixed << setprecision(4) << timer.elapsed() << "s" << endl;
}

data_t ClassifyNet::inference(const map<string, data_t>& data, int batch_size) {
    if (!net_initialized_) {
        cout << "net need to be initialized first!" << endl;
        exit(1);
    }
    if (data.find("img") == data.end()) {
        cout << "img data must be specified!" << endl;
        exit(1);
    }

    Timer timer;
    vector<data_t> data_vec {data.at("img")};
    DataProvider infer(data_vec, false);
    data_t data_inference;

    int infer_steps = infer.num_samples() / batch_size;
    int size_remain = infer.num_samples() % batch_size;
    for (int step = 0; step < infer_steps; ++step) {
        infer.load_batch({key_chunks_["img"]}, batch_size);
        key_chunks_["label"]->reshape(batch_size, 1, 1, 1);
        forward(false);
        const float* argmax_data = key_chunks_["argmax"]->const_data();
        for (int i = 0; i < batch_size; ++i) {
            data_inference.push_back({argmax_data[i]});
        }
    }
    if (size_remain) {
        infer.load_batch({key_chunks_["img"]}, size_remain);
        key_chunks_["label"]->reshape(size_remain, 1, 1, 1);
        forward(false);
        const float* argmax_data = key_chunks_["argmax"]->const_data();
        for (int i = 0; i < size_remain; ++i) {
            data_inference.push_back({argmax_data[i]});
        }
    }
    cout << "time used: " << timer.elapsed() << " s" << endl;

    return data_inference;
}

void ClassifyNet::forward(bool is_train, const string& layer_prefix) {
    //Timer t1;
    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        Timer timer;
        (*layer)->forward(is_train);
        //if (iter_ % 100 == 0)
        //    cout << (*layer)->layer_name_ << endl;
        layer_op_time_[(*layer)->layer_name_].first = timer.elapsed()*1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "forward: " << t1.elapsed()*1000 << endl;
    //}
}

void ClassifyNet::backward(const string& layer_prefix) {
    //Timer t1;
    for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
        Timer timer;
        (*layer)->backward();
        //if (iter_ % 100 == 0)
        //    cout << (*layer)->layer_name_ << endl;
        layer_op_time_[(*layer)->layer_name_].second = timer.elapsed()*1000;
    }
    //if (iter_ % 100 == 0) {
    //    cout << "backward: " << t1.elapsed()*1000 << endl;
    //}
}

void ClassifyNet::update(const string& layer_prefix) {
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
