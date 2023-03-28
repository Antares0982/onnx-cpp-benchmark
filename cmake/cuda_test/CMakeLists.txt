# --- find cuda

add_library(cudaOptions INTERFACE)

if (NOT WIN32)
    find_program(_nvidia_smi "nvidia-smi")
    if (_nvidia_smi)
        set(DETECT_GPU_COUNT_NVIDIA_SMI 0 PARENT_SCOPE)
        # execute nvidia-smi -L to get a short list of GPUs available
        execute_process(COMMAND ${_nvidia_smi} -L
                OUTPUT_VARIABLE _nvidia_smi_out
                RESULT_VARIABLE _nvidia_smi_ret)
        #    exec_program(${_nvidia_smi_path} ARGS -L
        #            OUTPUT_VARIABLE _nvidia_smi_out
        #            RETURN_VALUE    _nvidia_smi_ret)
        # process the stdout of nvidia-smi
        if (_nvidia_smi_ret EQUAL 0)
            # convert string with newlines to list of strings
            string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")
            foreach (_line ${_nvidia_smi_out})
                if (_line MATCHES "^GPU [0-9]+:")
                    math(EXPR DETECT_GPU_COUNT_NVIDIA_SMI "${DETECT_GPU_COUNT_NVIDIA_SMI}+1")
                    # the UUID is not very useful for the user, remove it
                    string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")
                    if (NOT _gpu_info STREQUAL "")
                        list(APPEND DETECT_GPU_INFO "${_gpu_info}")
                    endif ()
                endif ()
            endforeach ()

            set(DETECT_GPU_COUNT ${DETECT_GPU_COUNT_NVIDIA_SMI} PARENT_SCOPE)
            message("Detected GPU: ${DETECT_GPU_COUNT_NVIDIA_SMI}")
            if (${DETECT_GPU_COUNT_NVIDIA_SMI} GREATER 0)
                message("Add CUDA_ENABLED macro")
                target_compile_definitions(cudaOptions INTERFACE CUDA_ENABLED)
            endif ()
        else ()
            message(WARNING ${_nvidia_smi_out})
        endif ()
    else ()
        message("NVIDIA not supported")
    endif ()
    target_link_libraries(cudaOptions INTERFACE onnxruntime)
else ()
    message("Windows detected, skip CUDA detection")
endif ()
