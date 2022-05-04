NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-5;
    uint64_t iNetMiniBatch = 0;

    virtual void ValueAssign(NetBase &netSrc) { dAcc = netSrc.dAcc; iNetMiniBatch = netSrc.iNetMiniBatch; }
    void ShowIter() {}
    bool IterFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    virtual void ValueCopy(NetBase &netSrc) { ValueAssign(netSrc); }
    virtual void ValueMove(NetBase &&netSrc) { ValueAssign(netSrc); }
    NetBase(NetBase &netSrc) { ValueCopy(netSrc); }
    void operator=(NetBase &netSrc) { ValueCopy(netSrc); }
    
    NetBase(double dNetAcc = 1e-5, uint64_t iMinibatch = 0) : dAcc(dNetAcc), iNetMiniBatch(iMinibatch) {}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return 0; }
    bool Run() { return true; }
    void Reset() {}
    ~NetBase() { Reset(); }
};



NEUNET_END