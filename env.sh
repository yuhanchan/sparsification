# run source env.sh to set up ENV variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_HOME=$SCRIPT_DIR
echo "set PROJECT_HOME to: $PROJECT_HOME"