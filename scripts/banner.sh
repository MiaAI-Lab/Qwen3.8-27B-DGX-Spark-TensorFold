# The start.sh banner: "Qwen3.8-27B" in block letters (a terminal of 70+ columns), else one plain line.
banner() {
  local cols; cols=$(tput cols 2>/dev/null || echo 80)
  if [[ -t 1 && "$cols" -ge 69 ]]; then
    printf '  \033[1;38;2;255;150;80m%s\033[0m\n' ' ███                    ████         ███         ███  █████ ████'
    printf '  \033[1;38;2;249;134;100m%s\033[0m\n' '█   █                       █       █   █       █   █     █ █   █'
    printf '  \033[1;38;2;243;118;120m%s\033[0m\n' '█   █ █   █  ███  █ ██      █       █   █           █    █  █   █'
    printf '  \033[1;38;2;237;102;140m%s\033[0m\n' '█   █ █ █ █ █   █ ██  █  ███         ███  █████   ██    █   ████'
    printf '  \033[1;38;2;215;95;180m%s\033[0m\n' '█ █ █ █ █ █ █████ █   █     █       █   █        █     █    █   █'
    printf '  \033[1;38;2;190;95;215m%s\033[0m\n' '█  █  ██ ██ █     █   █     █  ██   █   █       █      █    █   █'
    printf '  \033[1;38;2;165;95;255m%s\033[0m\n' ' ██ █ █   █  ███  █   █ ████   ██    ███        █████  █    ████'
    printf '\033[2m  one DGX Spark · TensorFold %s · FP8 KV cache · DFlash2\033[0m\n' "${TF_VERSION:-v0.6.0}"
  else
    printf 'Qwen3.8-27B · one DGX Spark · TensorFold %s\n' "${TF_VERSION:-v0.6.0}"
  fi
}
