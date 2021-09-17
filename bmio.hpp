BMIO_BEGIN

class bitmap final
{
private:
    BMIO_CHANN R;
    BMIO_CHANN G;
    BMIO_CHANN B;
    GDI_STARTUP_INPUT st_gph;
    GDI_TOKEN gph_token;
    BMIO_STR dir_path(BMIO_STR dir_path_root, BMIO_STR name, char div_syb = '\\')
    {
        if(div_syb == '\\') return dir_path_root + '\\' + name + ".bmp";
        else if(div_syb == '/') return dir_path_root + '/' + name + ".bmp";
        else return "";
    }
    void init_chann(uint64_t ln_cnt, uint64_t col_cnt)
    {
        R = BMIO_CHANN(ln_cnt, col_cnt);
        G = BMIO_CHANN(ln_cnt, col_cnt);
        B = BMIO_CHANN(ln_cnt, col_cnt);
    }
    bool set_chann(BMIO_CHANN &R_src, BMIO_CHANN &G_src, BMIO_CHANN &B_src)
    {
        if(R_src.shape_valid(G_src) && G_src.shape_valid(B_src))
        {
            R = R_src;
            G = G_src;
            B = B_src;
            return true;
        }
        else return false;
    }
    bool CLSID_encode(const WCHAR* format, CLSID* p_CLSID)
    {
        // Number of image encoders
        uint32_t  num = 0;
        // Size of the image encoder array in bytes
        uint32_t  size = 0;
        Gdiplus::GetImageEncodersSize(&num, &size);
        if(size)
        {
            auto img_code_info_ptr = new Gdiplus::ImageCodecInfo[size];
            if(img_code_info_ptr)
            {
                Gdiplus::GetImageEncoders(num, size, img_code_info_ptr);
                for(auto i=0; i<num; ++i)
                {
                    if(!wcscmp(img_code_info_ptr[i].MimeType, format))
                    {
                        *p_CLSID = img_code_info_ptr[i].Clsid;
                        delete [] img_code_info_ptr;
                        return true;
                    }
                }
            }
            delete [] img_code_info_ptr;
        }
        return false;
    }
public:
    bitmap() {}
    
    ~bitmap() {}
};

BMIO_END