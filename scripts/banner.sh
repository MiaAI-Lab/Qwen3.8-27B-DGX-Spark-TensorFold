# The start.sh banner: "Qwen3.8-27B" in block letters (a terminal of 70+ columns), else one plain line.
banner() {
  local cols; cols=$(tput cols 2>/dev/null || echo 80)
  if [[ -t 1 && "$cols" -ge 69 ]]; then
    printf '  \033[1;38;2;205;240;50m%s\033[0m\n' ' ███                    ████         ███         ███  █████ ████'
    printf '  \033[1;38;2;165;230;60m%s\033[0m\n' '█   █                       █       █   █       █   █     █ █   █'
    printf '  \033[1;38;2;120;220;80m%s\033[0m\n' '█   █ █   █  ███  █ ██      █       █   █           █    █  █   █'
    printf '  \033[1;38;2;80;210;110m%s\033[0m\n' '█   █ █ █ █ █   █ ██  █  ███         ███  █████   ██    █   ████'
    printf '  \033[1;38;2;50;200;140m%s\033[0m\n' '█ █ █ █ █ █ █████ █   █     █       █   █        █     █    █   █'
    printf '  \033[1;38;2;40;195;170m%s\033[0m\n' '█  █  ██ ██ █     █   █     █  ██   █   █       █      █    █   █'
    printf '  \033[1;38;2;40;190;195m%s\033[0m\n' ' ██ █ █   █  ███  █   █ ████   ██    ███        █████  █    ████'
    printf '\033[2;38;2;120;220;140m  one DGX Spark · TensorFold %s · FP8 KV cache · DFlash2\033[0m\n' "${TF_VERSION:-v0.6.0}"
  else
    printf 'Qwen3.8-27B · one DGX Spark · TensorFold %s\n' "${TF_VERSION:-v0.6.0}"
  fi
}
