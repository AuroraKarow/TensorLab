CONV_BEGIN

struct ConvBN
{
    feature vecMiuBeta;
    feature vecSigmaSqr;
    bagrt::net_queue<feature> setBarX;
    bagrt::net_queue<feature> setY;
    ConvBN(){}
    ConvBN(ConvBN &ConvBNVal) {*this = ConvBNVal;}
    ConvBN(ConvBN &&ConvBNVal) {*this = std::move(ConvBNVal);}
    void operator=(ConvBN &ConvBNVal)
    {
        vecMiuBeta = ConvBNVal.vecMiuBeta;
        vecSigmaSqr = ConvBNVal.vecSigmaSqr;
        setBarX = ConvBNVal.setBarX;
        setY = ConvBNVal.setY;
    }
    void operator=(ConvBN &&ConvBNVal)
    {
        vecMiuBeta = std::move(ConvBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(ConvBNVal.vecSigmaSqr);
        setBarX = std::move(ConvBNVal.setBarX);
        setY = std::move(ConvBNVal.setY);
    }
    // ~ConvBN() {}
};

CONV_END