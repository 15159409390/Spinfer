export MAMBA_ROOT_PREFIX="/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER"
__mamba_setup="$("/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER/bin/mamba" shell hook --shell posix 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER/bin/mamba"  # Fallback on help from mamba activate
fi
unset __mamba_setup
