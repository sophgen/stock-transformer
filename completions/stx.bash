# Bash completion for stx (generated via Click: `_STX_COMPLETE=bash_source stx`)
# Install: copy to /etc/bash_completion.d/stx or source from ~/.bashrc
#
# Regenerate after CLI changes:
#   uv run stx completion bash > completions/stx.bash
#
# Or use eval in shell profile:
#   eval "$(uv run stx completion bash)"

_stx_completion() {
    local IFS=$'\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _STX_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

_stx_completion_setup() {
    complete -o nosort -F _stx_completion stx
}

_stx_completion_setup;
