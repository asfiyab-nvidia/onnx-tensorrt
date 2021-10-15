/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "NvInferRuntimeCommon.h"
#include "onnx2trt_utils.hpp"
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

namespace onnx2trt
{

//!
//! A simple implementation of the IErrorRecorder interface for
//! use by ONNX importer.
//! ONNX-importer Error recorder is based on a vector that pairs the error
//! code and the error string into a single element. It also uses
//! standard mutex and atomics in order to make sure that the code
//! works in a multi-threaded environment.
//!
class ONNXParserErrorRecorder : public nvinfer1::IErrorRecorder
{
    using RefCount       = nvinfer1::IErrorRecorder::RefCount;
    using ErrorDesc      = nvinfer1::IErrorRecorder::ErrorDesc;
    using ErrorCode      = nvinfer1::ErrorCode;
    using IErrorRecorder = nvinfer1::IErrorRecorder;
    using ILogger        = nvinfer1::ILogger;

    using errorPair      = std::pair<ErrorCode, std::string>;
    using errorStack     = std::vector<errorPair>;

public:
    static ONNXParserErrorRecorder* create(
        ILogger* logger, IErrorRecorder* otherRecorder = nullptr);

    static void destroy(ONNXParserErrorRecorder*& recorder);

    void     clear()       noexcept final;
    RefCount incRefCount() noexcept final;
    RefCount decRefCount() noexcept final;
    bool     reportError(ErrorCode val, ErrorDesc desc) noexcept final;

    int32_t getNbErrors() const noexcept final
    {
        return mErrorStack.size();
    }

    ErrorCode getErrorCode(int32_t errorIdx) const noexcept final
    {
        return invalidIndexCheck(errorIdx) ? ErrorCode::kINVALID_ARGUMENT : (*this)[errorIdx].first;
    }

    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept final
    {
        return invalidIndexCheck(errorIdx) ? "errorIdx out of range." : (*this)[errorIdx].second.c_str();
    }

    bool hasOverflowed() const noexcept final
    {
        // This class can never overflow since we have dynamic resize via std::vector usage.
        return false;
    }

protected:
    ONNXParserErrorRecorder(ILogger* logger, IErrorRecorder* otherRecorder = nullptr);

    virtual ~ONNXParserErrorRecorder() noexcept;

    static void logError(ILogger* logger, const char* str);

    // Simple helper functions.
    const errorPair& operator[](size_t index) const noexcept
    {
        return mErrorStack[index];
    }

    bool invalidIndexCheck(int32_t index) const noexcept
    {
        // By converting signed to unsigned, we only need a single check since
        // negative numbers turn into large positive greater than the size.
        size_t sIndex = index;
        return sIndex >= mErrorStack.size();
    }
    // Mutex to hold when locking mErrorStack.
    std::mutex mStackLock;

    // Reference count of the class. Destruction of the class when mRefCount
    // is not zero causes undefined behavior.
    std::atomic<int32_t> mRefCount{0};

    // The error stack that holds the errors recorded by TensorRT.
    errorStack mErrorStack;

    // Original error recorder (set by user)
    IErrorRecorder* mUserRecorder{nullptr};

    // logger
    ILogger* mLogger{nullptr};
}; // class ONNXParserErrorRecorder

} // namespace onnx2trt
