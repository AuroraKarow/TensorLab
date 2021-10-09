NEUNET_BEGIN

struct Layer
{
    uint64_t iActFuncType = SIGMOID;
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer() {}
    Layer(uint64_t iActFuncIdx) : iActFuncType(iActFuncIdx) {}
};

class LayerFCN : public Layer
{
private:
    vect vecWeight;
public:
    LayerFCN(uint64_t iActFuncIdx = SIGMOID) : Layer(iActFuncIdx) {}
    bool SetInputOutputLnCnt(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecWeight = fc::InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFCN(uint64_t iActFuncIdx, uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iActFuncIdx) {SetInputOutputLnCnt(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput) {return fc::Output(setInput, vecWeight);}
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = fc::GradLossToInput(setGrad, vecWeight);
        vecWeight -= dLearnRate * fc::GradLossToWeight(setGrad, setInput);
        return setGradBack;
    }
};



NEUNET_END