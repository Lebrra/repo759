#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
    int N = atoi(argv[1]);
    //int N = -1;   // kept these here for local testing
    //cin >> N;

    cout << "Received N = " << N << "\n";

    // print 0 - N:
    printf("printf forwards: \n");
    for (int i = 0; i <= N; i++) {
        printf("%i ", i);
    }

    // print N - 0:
    cout << "\ncout backwards: \n";
    for (int i = N; i >= 0; i--) {
        cout << i << " ";
    }
}