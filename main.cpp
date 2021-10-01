#include "bmio"

using namespace std;
using namespace bmio;
using namespace mtx;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    matrix test = {{1, 2},
                   {3, 4}};
    cout << test << endl;
    test.pos_idx(1) = 0;
    cout << test << endl;
    string root_dir = ".../TensorLab/";
    bitmap map(root_dir + "Src/Alice.jpg");
    map.save_img(root_dir + "Src", "Alice", BMIO_BMP, '/');
    return EXIT_SUCCESS;
}