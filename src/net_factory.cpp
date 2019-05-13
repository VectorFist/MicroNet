#include "net_factory.h"

namespace micronet {

Net build_resnet18(const chunk_ptr& img) {
    auto conv1 = Convolution(3, 3, 1, 1, 1, 1, 64)(img);
}

} // namespace micronet
