#ifndef NET_FACTORY_H_INCLUDED
#define NET_FACTORY_H_INCLUDED

#include "net.h"
#include "convolution.h"
#include "activation.h"
#include "pooling.h"
#include "add.h"

namespace micronet {

Net build_resnet18(const chunk_ptr& img);

} // namespace micronet

#endif // NET_FACTORY_H_INCLUDED
