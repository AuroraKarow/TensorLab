NEUNET_BEGIN

class Basnet
{
protected:
    layer_list lsLayer;
    virtual set<vect> ForwProp(set<vect> setInput)
    {
        
    }
    virtual void BackProp(set<vect> setGrad)
    {}
public:
    uint64_t Depth() {return lsLayer.size();}
    Basnet() {}
    Basnet(Basnet &netSrc) {*this = netSrc;}
    Basnet(Basnet &&netSrc) {*this = std::move(netSrc);}
    void operator=(Basnet &netSrc) {lsLayer = netSrc.lsLayer;}
    void operator=(Basnet &&netSrc) {lsLayer = std::move(netSrc.lsLayer);}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool AddLayer(Args&& ... pacArgs) {return lsLayer.emplace_back(pacArgs...);}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>>bool InsertLayer(uint64_t iIdx, Args&& ... pacArgs) {return lsLayer.insert(iIdx, pacArgs...);}
    virtual void Run()
    {}
};

NEUNET_END