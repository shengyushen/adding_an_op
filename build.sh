#bazel build --config opt zero_out_op_kernel_1.so
#bazel build --config opt cuda_op_kernel.so
bazel build --config opt bf16cut_fp.so
bazel build --config opt bf16cut_bp.so
bazel build --config opt bf16cut_bp_grad.so



