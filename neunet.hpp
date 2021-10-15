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
    uint64_t iNetMiniBatch = 1;
    bool ForwProp(set<vect> &setInputVec, set<feature> &setInputFt, bool bDeduce = false) {return true;}
    bool BackProp(set<vect> &setOrigin){return true;}
    // Call this function after back propagation
    virtual bool IterateFlag(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin) {return true;}
    void ValueAssign(Basnet &netSrc)
    {
        dNetAcc = netSrc.dNetAcc;
        bLastIsVect = netSrc.bLastIsVect;
        dNetLearnRate = netSrc.dNetLearnRate;
        iNetMiniBatch = netSrc.iNetMiniBatch;
    }
public:
    uint64_t Depth() {return lsLayer.size();}
    Basnet(double dAcc = 1e-5, double dLearnRate = 1e-10, uint64_t iMiniBatch = 0) : dNetAcc(dAcc), dNetLearnRate(dLearnRate), iNetMiniBatch(iMiniBatch) {}
    void SetInput(set<vect> setInput) {}
    Basnet(Basnet &netSrc)
    {
        lsLayer = netSrc.lsLayer;
        setOutputVec = netSrc.setOutputVec;
        setOutputFt = netSrc.setOutputFt;
        origin = netSrc.origin;
        setGradVec = netSrc.setGradVec;
        setGradFt = netSrc.setGradFt;
        ValueAssign(netSrc);
    }
    Basnet(Basnet &&netSrc)
    {
        lsLayer = std::move(netSrc.lsLayer);
        setOutputVec = std::move(netSrc.setOutputVec);
        setOutputFt = std::move(netSrc.setOutputFt);
        origin = std::move(netSrc.origin);
        setGradVec = std::move(netSrc.setGradVec);
        setGradFt = std::move(netSrc.setGradFt);
        ValueAssign(netSrc);
    }
    void operator=(Basnet &netSrc) {new(this)Basnet(netSrc);}
    void operator=(Basnet &&netSrc) {new(this)Basnet(std::move(netSrc));}
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
    virtual bool Run(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin) {return true;}
};

NEUNET_END