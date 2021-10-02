NEUNET_FC_BEGIN

vect InitWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    if(iInputLnCnt && iOutputLnCnt) return vect(iOutputLnCnt, iInputLnCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    else return blank_vect;
}

vect Output(vect &vecInput, vect &vecWeight) {return vecWeight * vecInput;}

vect GradLossToInput(vect &vecGradLossToOutput, vect &vecWeight) {return vecWeight.transposition() * vecGradLossToOutput;}

vect GradLossToWeight(vect &vecGradLossToOutput, vect &vecInput) {return vecGradLossToOutput * vecInput.transposition();}

vect FeatureTransform(feature &vecInput)
{
    auto iElemCnt = vecInput[0].ELEM_CNT,
        iChannCnt = vecInput.size(),
        iLnCnt = iChannCnt * iElemCnt,
        iCpyCnt = 0Ui64;
    auto vecResTransForm = vect(iLnCnt, 1);
    for(auto i=0; i<iChannCnt; ++i) for(auto j=0; j<iElemCnt; ++j) vecResTransForm.pos_idx(++ iCpyCnt) = vecInput[i].pos_idx(j);
    return vecResTransForm;
}

feature FeatureTransform(vect &vecInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    auto iElemCnt = iLnCnt * iColCnt,
        iChannCnt = vecInput.LN_CNT / iElemCnt,
        iCpyCnt = 0Ui64;
    feature vecResTransForm(iChannCnt);
    for(auto i=0; i<iChannCnt; ++i) for(auto j=0; j<iElemCnt; ++j) vecResTransForm[i].pos_idx(j) = vecInput.pos_idx(iCpyCnt++);
    return vecResTransForm;
}

NEUNET_FC_END