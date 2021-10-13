NEUNET_BEGIN

class Basnet
{
protected:
    LAYER_LIST lsLayer;
    set<vect> setOutputVec;
    set<feature> setOutputFt;
    set<vect> origin;
    double dNetAcc = 1e-5;
    bool bLastIsVect = true;
    bool ForwProp(set<vect> &setInputVec, set<feature> &setInputFt)
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
                setOutputFt = LAYER_INSTANCE<LAYER_CONV>(head->data) -> ForwProp(setInputFt);
                if(setOutputFt.size()) break;
                else return false;
            case CONV_ADA:
                setOutputFt = LAYER_INSTANCE<LAYER_CONV_ADA>(head->data) -> ForwProp(setInputFt);
                if(setOutputFt.size()) break;
                else return false;
            case POOL:
                setOutputFt = LAYER_INSTANCE<LAYER_POOL>(head->data) -> ForwProp(setInputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_FC:
                setOutputVec = LAYER_INSTANCE<LAYER_BN_FC>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_FC_ADA:
                setOutputVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_CONV:
                setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV>(head->data) -> ForwProp(setInputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_CONV_ADA:
                setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(head->data) -> ForwProp(setInputFt);
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
        set<vect> setGradVec;
        set<feature> setGradFt;
        if(bLastIsVect) {setGradVec = softmax_cec_grad(setOutputVec, setOrigin);}
        else setGradVec = cec_grad(setOutputVec, setOrigin);
        auto tail_node = lsLayer.tail_node();
        while(tail_node)
        {}
        return true;
    }
public:
    uint64_t Depth() {return lsLayer.size();}
    Basnet(double dAcc = 1e-5) : dNetAcc(dAcc) {}
    void SetInput(set<vect> setInput) {}
    Basnet(Basnet &netSrc) {*this = netSrc;}
    Basnet(Basnet &&netSrc) {*this = std::move(netSrc);}
    void operator=(Basnet &netSrc) {lsLayer = netSrc.lsLayer;}
    void operator=(Basnet &&netSrc) {lsLayer = std::move(netSrc.lsLayer);}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool AddLayer(Args&& ... pacArgs) {return lsLayer.emplace_back(pacArgs...);}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool InsertLayer(uint64_t iIdx, Args&& ... pacArgs) {return lsLayer.insert(iIdx, pacArgs...);}
    void Run()
    {}
};

NEUNET_END