NEUNET_FC_BEGIN

vect InitWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    if(iInputLnCnt && iOutputLnCnt) return vect(iOutputLnCnt, iInputLnCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    else return blank_vect;
}

vect Output(vect &vecInput, vect &vecWeight) {return vecWeight * vecInput;}

vect GradLossToInput(vect &vecGradLossToOutput, vect &vecWeight) {return vecWeight.transposition() * vecGradLossToOutput;}

vect GradLossToWeight(vect &vecGradLossToOutput, vect &vecInput) {return vecGradLossToOutput * vecInput.transposition();}



NEUNET_FC_END