# Source on every DGX Spark before `ray start` and before the FastH3 driver.
# QSFP ConnectX-7 interface names match the GB10 dual-Spark bring-up
# (enp1s0f1np1 / rocep1s0f1). Override NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME /
# NCCL_IB_HCA if `ibdev2netdev` shows different names.
#
#   source examples/inference/optimizations/spark_pair_env.sh
#   export FASTVIDEO_HOST_IP=<this node's QSFP IPv4>
#
# See docs/getting_started/installation/spark_pair.md

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp1s0f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}"
# GB10 has no NVLink between boxes. Intra-node C2C P2P fights the QSFP path.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# Ray's default memory monitor treats GB10 unified RSS during DiT load as a
# runaway and SIGTERMs the worker around shard 11/14.
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-1.0}"
