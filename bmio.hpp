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
    bool init_chann(uint64_t ln_cnt, uint64_t col_cnt)
    {
        if(ln_cnt && col_cnt)
        {
            R = BMIO_CHANN(ln_cnt, col_cnt);
            G = BMIO_CHANN(ln_cnt, col_cnt);
            B = BMIO_CHANN(ln_cnt, col_cnt);
            return true;
        }
        else return false;
    }
    bool set_chann(BMIO_CHANN &R_src, BMIO_CHANN &G_src, BMIO_CHANN &B_src, bool move_flag = false)
    {
        if(R_src.shape_valid(G_src) && G_src.shape_valid(B_src))
        {
            if(move_flag)
            {
                R = std::move(R_src);
                G = std::move(G_src);
                B = std::move(B_src);
            }
            else
            {
                R = R_src;
                G = G_src;
                B = B_src;
            }
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
                        img_code_info_ptr = nullptr;
                        return true;
                    }
                }
            }
            delete [] img_code_info_ptr;
            img_code_info_ptr = nullptr;
        }
        return false;
    }
public:
    bitmap() {}
    bool set_size(uint64_t ln_cnt, uint64_t col_cnt){return init_chann(ln_cnt, col_cnt);}
    bool set_raw(BMIO_CHANN &R_src, BMIO_CHANN &G_src, BMIO_CHANN &B_src) {return set_chann(R_src, G_src, B_src, true);}
    uint64_t ln_cnt() {return R.LN_CNT;}
    uint64_t col_cnt() {return R.COL_CNT;}
    __declspec (property (get=ln_cnt)) uint64_t HEIGHT;
    __declspec (property (get=col_cnt)) uint64_t WIDTH;
    bool img_valid() {return HEIGHT && WIDTH && R.shape_valid(G) && G.shape_valid(B);}
    bitmap(bitmap &val) {*this = val;}
    bitmap(bitmap &&val) {*this = std::move(val);}
    bitmap(BMIO_RAW &&vec) {if(vec.size() == BMIO_RGB_CNT) set_chann(vec[BMIO_R], vec[BMIO_G], vec[BMIO_B], true);}
    bitmap(BMIO_RAW &vec) {if(vec.size() == BMIO_RGB_CNT) set_chann(vec[BMIO_R], vec[BMIO_G], vec[BMIO_B]);}
    bool load_img(BMIO_WSTR dir)
    {
        GDI_STARTUP(&gph_token, &st_gph, nullptr);
        auto map_ptr = new GDI_BITMAP(dir.c_str());
        bool valid_flag = true;
        if(map_ptr)
        {
            init_chann(map_ptr->GetHeight(), map_ptr->GetWidth());
            for(uint64_t i=0; i<map_ptr->GetHeight(); i++)
                for(uint64_t j=0; j<map_ptr->GetWidth(); j++)
                {
                    GDI_COLOR color;
                    map_ptr->GetPixel(i, j, &color);
                    R[i][j] = color.GetRed();
                    G[i][j] = color.GetGreen();
                    B[i][j] = color.GetBlue();
                }
        }
        else valid_flag = false;
        delete map_ptr;
        map_ptr = nullptr;
        GDI_SHUTDOWN(gph_token);
        return valid_flag;
    }
    bool load_img(BMIO_STR dir) {return load_img(BMIO_CHARSET(dir));}
    bitmap(BMIO_STR dir) {load_img(dir);}
    bitmap(BMIO_WSTR dir) {load_img(dir);}
    bool save_img(BMIO_WSTR dir_root, BMIO_WSTR name, uint64_t extend, wchar_t div_syb = L'\\')
    {
        if(img_valid())
        {
            GDI_STARTUP(&gph_token, &st_gph, nullptr);
            GDI_BITMAP bitmap(WIDTH, HEIGHT, PixelFormat24bppRGB);
            GDI_GRAPHICS gph_img(&bitmap);
            CLSID CID_STR;
            BMIO_WSTR ext_name = L"";
            switch (extend)
            {
            case BMIO_PNG:
                if(CLSID_encode(L"image/png", &CID_STR))
                {
                    ext_name = L".png";
                    break;
                }
                else return false;
            case BMIO_JPG:
                if(CLSID_encode(L"image/jpeg", &CID_STR))
                {
                    ext_name = L".jpg";
                    break;
                }
                else return false;
            case BMIO_GIF:
                if(CLSID_encode(L"image/gif", &CID_STR))
                {
                   ext_name = L".gif";
                   break; 
                }
                else return false;
            case BMIO_TIF:
                if(CLSID_encode(L"image/tiff", &CID_STR))
                {
                    ext_name = L".tif";
                    break;
                }
                else return false;
            case BMIO_BMP:
                if(CLSID_encode(L"image/bmp", &CID_STR))
                {
                    ext_name = L".bmp";
                    break;
                }
                else return false;
            default: return false;
            }
            // Draw image
            for(auto i=0; i<HEIGHT; ++i)
                for(auto j=0; j<WIDTH; ++j)
                    gph_img.DrawLine(&GDI_PEN(GDI_COLOR(R[i][j], G[i][j], B[i][j])), i, j, i+1, j+1);
            BMIO_WSTR path = dir_root, file_name = name + ext_name;
            if(div_syb == L'\\') path += L'\\' + file_name;
            else path += L'/' + file_name;
            return bitmap.Save(path.c_str(), &CID_STR) == GDI_STATUS::Ok;
        }
        else return false;
    }
    bool save_img(BMIO_STR dir_root, BMIO_STR name, uint64_t extend, char div_syb = '\\') {return save_img(BMIO_CHARSET(dir_root), BMIO_CHARSET(name), extend, div_syb);}
    BMIO_CHANN gray()
    {
        if(img_valid()) return BMIO_GRAY_WEIGHT_R * R + BMIO_GRAY_WEIGHT_G * G + BMIO_GRAY_WEIGHT_B * B;
        else return BMIO_CHANN::blank_matrix();
    }
    static BMIO_RAW gray_grad(BMIO_CHANN &grad_vec)
    {
        if(grad_vec.is_matrix())
        {
            BMIO_RAW RGB_gradient(BMIO_RGB_CNT);
            RGB_gradient[BMIO_R] = grad_vec * BMIO_GRAY_WEIGHT_R;
            RGB_gradient[BMIO_G] = grad_vec * BMIO_GRAY_WEIGHT_G;
            RGB_gradient[BMIO_B] = grad_vec * BMIO_GRAY_WEIGHT_B;
            return RGB_gradient;
        }
        else return BMIO_RAW::blank_queue();
    }
    BMIO_RAW img_vec()
    {
        BMIO_RAW raw_vec(BMIO_RGB_CNT);
        for(auto i=0; i<BMIO_RGB_CNT; ++i)
        {
            raw_vec[BMIO_R] = R;
            raw_vec[BMIO_G] = G;
            raw_vec[BMIO_B] = B;
        }
        return raw_vec;
    }
    bool operator==(bitmap &val) {return R==val.R && G==val.G && B==val.B;}
    bool operator!=(bitmap &val) {return !(*this == val);}
    void operator=(bitmap &val) {set_chann(val.R, val.G, val. B);}
    void operator=(bitmap &&val) {set_chann(val.R, val.G, val. B, true);}
    ~bitmap() {}
};

BMIO_END