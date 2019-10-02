/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "ShapedWeights.hpp"
#include "trt_utils.hpp"

#include <onnx/onnx.pb.h>
#include <onnx/onnxifi.h>
#include <NvInfer.h>

#include <iostream>
#include <sstream>
#include <cstring>  // For std::memcpy
#include <numeric>

#define LOG_VERBOSE(msg)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        ss << __FILE__ << ":" << __LINE__ << ": " << msg;                                                              \
        ctx->logger().log(nvinfer1::ILogger::Severity::kVERBOSE, ss.str().c_str());                                    \
    } while (0)

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape) {
    stream << "(";
    for (int i = 0; i < shape.nbDims; ++i)
    {
        stream << (i ? ", " : "") << shape.d[i];
    }
    return stream << ")";
}

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype) {
  switch( dtype ) {
  case nvinfer1::DataType::kFLOAT: return stream << "float32";
  case nvinfer1::DataType::kHALF:  return stream << "float16";
  case nvinfer1::DataType::kINT8:  return stream << "int8";
  case nvinfer1::DataType::kINT32: return stream << "int32";
  case nvinfer1::DataType::kBOOL: return stream << "bool";
  default: throw std::runtime_error("Unknown dtype");
  }
}

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm) {
    int ndims = nvinfer1::Dims::MAX_DIMS;
    stream << "(" << perm.order[0];
    for (int i = 1; i < ndims; ++i)
    {
        stream << ", " << perm.order[i];
    }
    stream << ")";
    return stream;
}

namespace onnx2trt {

// Helper function to calculate the volume of a Dims object
int64_t volume(const nvinfer1::Dims& dims);

// Helper function to get the size in bytes of an ONNX datatype
int getDtypeSize(int32_t onnxDtype);

// Helper function to add a scalar into TRT through a constant layer.
template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstantScalar(IImporterContext* ctx, ScalarType scalar, ShapedWeights::DataType type, nvinfer1::Dims shape = nvinfer1::Dims{0})
{
    assert(volume(shape) == 1 && "Cannot add constant scalar with a shape that has volume > 1");
    ShapedWeights scalarWeights = ctx->createTempWeights(type, shape);
    static_cast<ScalarType*>(scalarWeights.values)[0] = scalar;
    return ctx->network()->addConstant(scalarWeights.shape, scalarWeights);
}

// Helper function to create a tensor given a vector of values and a shape.
template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstant(IImporterContext* ctx, const std::vector<ScalarType>& values, ShapedWeights::DataType type, nvinfer1::Dims shape)
{
    assert(volume(shape) == static_cast<int64_t>(values.size()) && "Shape does not match number of values provided");
    assert(sizeof(ScalarType) == getDtypeSize(type) && "ONNX dtype does not have the same size as the value type");
    ShapedWeights weights = ctx->createTempWeights(type, shape);
    std::memcpy(weights.values, values.data(), values.size() * sizeof(ScalarType));
    return ctx->network()->addConstant(weights.shape, weights);
}

// Helper class to change output dimension calculations for pooling operators
class CeilingPoolDim:public nvinfer1::IOutputDimensionsFormula
{
public:
    nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
        nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, nvinfer1::DimsHW dilation, const char* layerName) const
    {
        nvinfer1::DimsHW outputDims;
        for (int dimension = 0; dimension < inputDims.nbDims; ++dimension)
        {
            outputDims.d[dimension] = static_cast<int>(ceil((inputDims.d[dimension] + padding.d[dimension] * 2.0 - kernelSize.d[dimension]) / stride.d[dimension] + 1.0));
        }
        return outputDims;
    }
};

enum ScaleOp
{
    kSHIFT,
    kSCALE,
    kPOWER,
};

// Helper function to import ONNX activation nodes into TRT
NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha = nullptr, float* beta = nullptr);

// Helper function to import ArgMax and ArgMin nodes into TRT
NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op);

// Helper function to broadcast two tensors to the larger one's shape
void broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

// Helper function to broadcast one tensor to a given shape.
void broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, const int nbDims);

// Helper function for constantOfShape operator. Input shape must be a shape tensor
nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape);

// Helper function to convert an ONNX axis into a TRT axis
Status convertAxis(int& axis, int nbDims);

// Helper function to convert an ONNX datatype into a TRT datatype
bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert INT64 weight values into INT32
int32_t* convertINT64(const int64_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx);

// Helper function to convert an ONNX weight into a ShapedWeights object
bool convertOnnxWeights(const ::ONNX_NAMESPACE::TensorProto& onnxTensor, onnx2trt::ShapedWeights* weights, IImporterContext* ctx);

// Helper function to convert a 1D tensor into a scalar
nvinfer1::ITensor* convertToScalar(IImporterContext* ctx, nvinfer1::ITensor* inpTensor);

// Helper function to convert a ShapedWeights object into a tensor
nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx);

// Helper function to convert an ONNX weight descriptor into a ShapedWeights object
bool convertWeightDescriptor(onnxTensorDescriptorV1 const &desc, onnx2trt::ShapedWeights *weights, IImporterContext* ctx);

// Helper function to provide a ceiling-rounding division between two integers
int divCeil(int n, int d);

// Helper function to import an ONNX elementwise op into TRT
NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op);

// Helper function to flatten a tensor on a given axis
nvinfer1::ITensor* flattenTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis = 0);

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims input_dims, nvinfer1::Dims output_shape, nvinfer1::Dims kernel_size,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& beg_padding,
    nvinfer1::Dims& end_padding, nvinfer1::Dims& output_padding, nvinfer1::PaddingMode paddingMode);

// Helper function to get default ONNX activation alpha values
float getActivationDefaultAlpha(nvinfer1::ActivationType type);

// Helper function to get default ONNX activation beta values
float getActivationDefaultBeta(nvinfer1::ActivationType type);

// Helper function to get the length of the specified axis
nvinfer1::ITensor* getAxisLength(IImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape=nvinfer1::Dims{0});

// Helper function to calculate the output size of a convolution node given its attributes
int getConvOutputSize(int input_size, int filter_size,
                                int stride, int dilation_rate,
                                int total_padding);

// Helper function to get the TRT datatype given an ONNX datatype
const char* getDtypeName(int32_t onnxDtype);

// Helper function to get kernel attributes for various ONNX nodes
void getKernelParams(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::Dims* kernel_size,
                       nvinfer1::Dims* strides,
                       nvinfer1::Dims* beg_padding,
                       nvinfer1::Dims* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       bool& count_exclude_padding,
                       nvinfer1::Dims* dilations=nullptr,
                       nvinfer1::Dims* output_padding=nullptr);

// Helper function to get the scaling mode for TRT's scale layer
nvinfer1::ScaleMode getScaleMode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape);

// Helper function to get a plugin from the PluginRegistry
nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields);

// Helper function to determine if a transpose is required
bool isTransposeRequired(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm);

// Helper function to import LSTM ops through the legacy CUDNN path
NodeImportResult lstmLegacyImporter(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);

// Helper function to create and fill a Dims object with defined values
nvinfer1::Dims makeDims(int nbDims, int val);

// Helper function to create a Shape Tensor from a Dims object
nvinfer1::ITensor& makeShapeTensor(IImporterContext* ctx, nvinfer1::Dims dims);

// Helper function to import reduce ops into TRT
NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation);

// Helper function to shape a Tensor given a new shape
nvinfer1::ITensor* reshapeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape);

// Helper function to map attributes to a TRT scale layer
NodeImportResult scaleHelper(IImporterContext* ctx, nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode,
    nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power);

// Helper function to set an ONNX attribute
void setAttr(nvinfer1::Dims * trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal);

// Helper function to transpose a tensor given a permutation
nvinfer1::ITensor* transposeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm, bool permute_dim_types = true);

// Helper function to import ONNX unary ops into TRT
NodeImportResult unaryHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::UnaryOperation op);

// Helper function to convert a ShapedWeights object into a vector
Status weightsToVector(TensorOrWeights weights, std::vector<int64_t>* weightVector);

} // namespace onnx2trt
