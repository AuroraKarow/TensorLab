NEUNET_BEGIN

struct Layer{};

class LayerFCN : public Layer
{
private:
    vect vecWeight;
public:
    LayerFCN() {}
    bool SetInputOutputLnCnt(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecWeight = fc::InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFCN(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) {SetInputOutputLnCnt(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
};

class LayerActivate
{};

NEUNET_END