#include "dataset"
#include "neunet"

using namespace std;
using namespace dataset;
using namespace neunet;

class NetMNIST final : public NetClassify
{
private:
    set<set<feature>> batInput;
public:
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", 20, true, true, 2);
    
    return EXIT_SUCCESS;
}