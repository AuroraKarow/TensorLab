#include "funclib"

using namespace std;
using namespace bagrt;
using namespace mtx;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    auto m = matrix({{1, 2},
                     {3, 4}});
    vect_t<int> test(10);
    return EXIT_SUCCESS;
}