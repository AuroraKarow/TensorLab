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
        if(bLastIsVect) {setGradVec = softmax_cec_grad(setOutputVec, setOrigin);}
        else setGradVec = cec_grad(setOutputVec, setOrigin);
        auto tail = lsLayer.tail_node();
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
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool AddLayer(Args&& ... pacArgs) {return lsLayer.emplace_back(pacArgs...);}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool InsertLayer(uint64_t iIdx, Args&& ... pacArgs) {return lsLayer.insert(iIdx, pacArgs...);}
    void Run(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin)
    {
        do
        {
            ForwProp(setInputVec, setInputFt);
            BackProp(setOrigin);
        } while (IterateFlag(setInputVec, setInputFt, setOrigin));
    }
};

NEUNET_END