NEUNET_BEGIN

class Basnet
{
protected:
    /* Network */
    LAYER_LIST lsLayer;
    /* Input */
    set<vect> setOutputVec;
    set<feature> setOutputFt;
    /* Gradient */
    set<vect> origin;
    set<vect> setGradVec;
    set<feature> setGradFt;
    /* Parameter */
    double dNetAcc = 1e-5;
    bool bLastIsVect = true;
    double dNetLearnRate = 0;
    uint64_t iNetMiniBatch = 0;
    bool ForwProp(set<vect> &setInputVec, set<feature> &setInputFt, bool bDeduce = false)
    {
        auto head = lsLayer.head_node();
        setOutputVec = setInputVec;
        setOutputFt = setInputFt;
        while(head)
        {
            switch (head->data->iLayerType)
            {
            case FC:
                setOutputVec = LAYER_INSTANCE<LAYER_FC>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case FC_ADA:
                setOutputVec = LAYER_INSTANCE<LAYER_FC_ADA>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case CONV:
                setOutputFt = LAYER_INSTANCE<LAYER_CONV>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case CONV_ADA:
                setOutputFt = LAYER_INSTANCE<LAYER_CONV_ADA>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case POOL:
                setOutputFt = LAYER_INSTANCE<LAYER_POOL>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_FC:
                if(bDeduce) setOutputVec = LAYER_INSTANCE<LAYER_BN_FC>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputVec = LAYER_INSTANCE<LAYER_BN_FC>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_FC_ADA:
                if(bDeduce) setOutputVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_CONV:
                if(bDeduce) setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_CONV_ADA:
                if(bDeduce) setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(head->data) -> Deduce(iNetMiniBatch);
                setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            default: return false;
            }
            head = head -> next();
        }
        return true;
    }
    bool BackProp(set<vect> &setOrigin)
    {
        auto tail = lsLayer.tail_node();
        if(bLastIsVect)  setGradVec = softmax_cec_grad(setOutputVec, setOrigin);
        else setGradVec = cec_grad(setOutputVec, setOrigin);
        while(tail)
        {
            switch (tail->data->iLayerType)
            {
            case FC:
                setGradVec = LAYER_INSTANCE<LAYER_FC>(tail->data) -> BackProp(setGradVec, dNetLearnRate);
                if(setGradVec.size()) break;
                else return false;
            case FC_ADA:
                setGradVec = LAYER_INSTANCE<LAYER_FC_ADA>(tail->data) -> BackProp(setGradVec);
                if(setGradVec.size()) break;
                else return false;
            case CONV:
                setGradFt = LAYER_INSTANCE<LAYER_CONV>(tail->data) -> BackProp(setGradFt, dNetLearnRate);
                if(setGradFt.size()) break;
                else return false;
            case CONV_ADA:
                setGradFt = LAYER_INSTANCE<LAYER_CONV_ADA>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            case POOL:
                setGradFt = LAYER_INSTANCE<LAYER_POOL>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            case BN_FC:
                setGradVec = LAYER_INSTANCE<LAYER_BN_FC>(tail->data) -> BackProp(setGradVec, dNetLearnRate);
                if(setGradVec.size()) break;
                else return false;
            case BN_FC_ADA:
                setGradVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(tail->data) -> BackProp(setGradVec);
                if(setGradVec.size()) break;
                else return false;
            case BN_CONV:
                setGradFt = LAYER_INSTANCE<LAYER_BN_CONV>(tail->data) -> BackProp(setGradFt, dNetLearnRate);
                if(setGradFt.size()) break;
                else return false;
            case BN_CONV_ADA:
                setGradFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            default: return false;
            }
            tail = tail -> prev();
        }
        return true;
    }
    // Call this function after back propagation
    bool IterateFlag(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin)
    {
        if(ForwProp(setInputVec, setInputFt, true))
        {
            for(auto i=0; i<setOrigin.size(); ++i)
                for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
                    if(std::abs(setOrigin[i][j][ZERO_IDX] - setOutputVec[i][j][ZERO_IDX]) > dNetAcc) return true;
        }
        return false;
    }
public:
    uint64_t Depth() {return lsLayer.size();}
    Basnet(double dAcc = 1e-5, double dLearnRate = 1e-10, uint64_t iMiniBatch = 0) : dNetAcc(dAcc), dNetLearnRate(dLearnRate), iNetMiniBatch(iMiniBatch) {}
    void SetInput(set<vect> setInput) {}
    Basnet(Basnet &netSrc) {*this = netSrc;}
    Basnet(Basnet &&netSrc) {*this = std::move(netSrc);}
    void operator=(Basnet &netSrc) {lsLayer = netSrc.lsLayer;}
    void operator=(Basnet &&netSrc) {lsLayer = std::move(netSrc.lsLayer);}
    /* Add Layer
     * - Parameter selector
     * iActFuncIdx
     * [SIGMOID, RELU, SOFTMAX]
     * iCurrLayerType
     * [FC, FC_ADA, CONV, CONV_ADA, POOL, BN_FC, BN_FC_ADA, BN_CONV, BN_CONV_ADA, TRANS_TO_VECT, TRANS_TO_FEAT]
     * iPoolIdx
     * [POOL_AVG, POOL_MAX, POOL_GAG]
     * - LAYER_FC
     * iInputLnCnt, iOutputLnCnt, iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC, bIsFirstLayer = false, dRandBoundryFirst = 0.0, dRandBoundrySecond = 0.0, dAcc = 1e-05
     * - LAYER_FC_ADA
     * iInputLnCnt, iOutputLnCnt, iActFuncIdx = SIGMOID, dDecayController = 0.95, dAdaDominator = 1e-6, iCurrLayerType = FC_ADA, bIsFirstLayer = false, dRandBoundryFirst = 0.0, dRandBoundrySecond = 0.0, dAcc = 1e-05
     * - LAYER_CONV
     * iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx = RELU, iCurrLayerType = CONV, bIsFirstLayer = false, dRandBoundryFirst = 0, dRandBoundrySecond = 0, dAcc = 1e-5, iLnDilation = 0, iColDilation = 0, iInputPadTop = 0, iInputPadRight = 0, iInputPadBottom = 0, iInputPadLeft = 0, iLnDistance = 0, iColDistance = 0
     * - LAYER_CONV_ADA
     * iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx = RELU, iCurrLayerType = CONV_ADA, bIsFirstLayer = false, dDecayController = 0.95, dRandBoundryFirst = 0, dRandBoundrySecond = 0, dAcc = 1e-5, dAdaDominator = 1e-6, iLnDilation = 0, iColDilation = 0, iInputPadTop = 0, iInputPadRight = 0, iInputPadBottom = 0, iInputPadLeft = 0, iLnDistance = 0, iColDistance = 0
     * - LAYER_POOL
     * iFilterLnCnt = 0, iFilterColCnt = 0, iLnStride = 0, iColStride = 0, iPoolIdx = POOL_MAX, iActFuncIdx = NULL_FUNC, iCurrLayerType = POOL, bIsFirstLayer = false, iLnDilation = 0, iColDilation
     * - LAYER_BN_FC
     * iActFuncIdx = NULL_FUNC, dShift = 0, dScale = 1, iCurrLayerType = BN_FC, bIsFirstLayer = false, dBNDominator = 1e-10
     * - LAYER_BN_FC_ADA
     * iActFuncIdx = NULL_FUNC, dShift = 0, dScale = 1, iCurrLayerType = BN_FC_ADA, bIsFirstLayer = false, dDecayController = 0.95, dAdaDominator = 1e-6, dBNDominator = 1e-10
     * - LAYER_BN_CONV
     * iChannCnt, iActFuncIdx = NULL_FUNC, dShift = 0, dScale = 1, iCurrLayerType = BN_CONV, bIsFirstLayer = false, dBNDominator = 1e-5
     * - LAYER_BN_CONV_ADA
     * iChannCnt = 1, iActFuncIdx = NULL_FUNC, dShift = 0, dScale = 1, iCurrLayerType = BN_CONV, bIsFirstLayer = false, dDecayController = 0.95, dAdaDominator = 1e-3, dBNDominator = 1e-10
     * - TRANS_TO_VECT
     * iActFuncIdx = NULL_FUNC, iCurrLayerType = TRANS_TO_VECT, isFirstLayer = false
     * - TRANS_TO_FEAT
     * iChannLnCnt, iChannColCnt, iActFuncIdx = NULL_FUNC, iCurrLayerType = TRANS_TO_FEAT, isFirstLayer = false
     */
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) {return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...));}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool InsertLayer(uint64_t iIdx, Args&& ... pacArgs) {return lsLayer.insert(iIdx, std::make_shared<LayerType>(pacArgs...));}
    bool Run(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin)
    {
        do if(!ForwProp(setInputVec, setInputFt) || !BackProp(setOrigin)) return false;
        while (IterateFlag(setInputVec, setInputFt, setOrigin));
        return true;
    }
};

NEUNET_END