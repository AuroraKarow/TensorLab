#include "bmio"

using namespace std;
using namespace bmio;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    bitmap map("D:/Users/Aurora/Documents/Visual Studio Code Project/TensorLab/Src/Alice.jpg");
    map.save_img("D:/Users/Aurora/Documents/Visual Studio Code Project/TensorLab/Src", "Alice", BMIO_BMP, '/');
    return EXIT_SUCCESS;
}