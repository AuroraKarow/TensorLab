FC_BEGIN

vect InitWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    if(iInputLnCnt && iOutputLnCnt) return vect(iOutputLnCnt, iInputLnCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    else return blank_vect;
}

set<vect> Output(set<vect> &setInput, vect &vecWeight)
{
    set<vect> setOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOutput[i] = Output(setInput[i], vecWeight);
        if(!setOutput[i].is_matrix()) return blank_vect_seq;
    }
    return setOutput;
}

set<vect> GradLossToInput(set<vect> &setGradLossToOutput, vect &vecWeight)
{
    set<vect> setGradLossToInput(setGradLossToOutput.size());
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
    {
        setGradLossToInput[i] = GradLossToInput(setGradLossToOutput[i], vecWeight);
        if(!setGradLossToInput[i].is_matrix()) return blank_vect_seq;
    }
    return setGradLossToInput;
}

set<vect> GradLossToWeightSet(set<vect> &setGradLossToOutput, set<vect> &setInput)
{
    set<vect> vecGradLossToWeight;
    if(setGradLossToOutput.size() == setInput.size())
    {
        vecGradLossToWeight.init(setInput.size());
        for(auto i=0; i<setInput.size(); ++i)
        {
            vecGradLossToWeight[i] = GradLossToWeight(setGradLossToOutput[i], setInput[i]);
            if(!vecGradLossToWeight[i].is_matrix()) return blank_vect_seq;
        }
    }
    return vecGradLossToWeight;
}

vect GradLossToWeight(set<vect> &setGradLossToOutput, set<vect> &setInput)
{
    auto setGradLossToWeight = GradLossToWeightSet(setGradLossToOutput, setInput);
    return setGradLossToWeight.sum();
}

set<vect> FeatureTransform(set<feature> &setInput)
{
    set<vect> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransform(setInput[i]);
        if(!setOuput[i].is_matrix()) return blank_vect_seq;
    }
    return setOuput;
}

set<feature> FeatureTransform(set<vect> &setInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    set<feature> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransform(setInput[i], iLnCnt, iColCnt);
        if(!setOuput[i].size()) return blank_ft_seq;
    }
    return setOuput;
}

struct FCBN
{
    vect vecMiuBeta;
    vect vecSigmaSqr;
    set<vect> setBarX;
    set<vect> setY;
    FCBN(){}
    FCBN(FCBN &FCBNVal) {*this = FCBNVal;}
    FCBN(FCBN &&FCBNVal) {*this = std::move(FCBNVal);}
    void operator=(FCBN &FCBNVal)
    {
        vecMiuBeta = FCBNVal.vecMiuBeta;
        vecSigmaSqr = FCBNVal.vecSigmaSqr;
        setBarX = FCBNVal.setBarX;
        setY = FCBNVal.setY;
    }
    void operator=(FCBN &&FCBNVal)
    {
        vecMiuBeta = std::move(FCBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(FCBNVal.vecSigmaSqr);
        setBarX = std::move(FCBNVal.setBarX);
        setY = std::move(FCBNVal.setY);
    }
    // ~FCBN() {}
};

FC_END