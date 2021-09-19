#include "bmio"

using namespace std;
using namespace bmio;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    bitmap map("E:/VS Code project data/TensorLab/Src/Alice.jpg");
    map.save_img("E:/VS Code project data/TensorLab/Src", "Alice", BMIO_BMP, '/');
    return EXIT_SUCCESS;
}