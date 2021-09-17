CSVIO_BEGIN

bagrt::net_queue<std::string> split_string(std::string string_value, char cond_syb){
    bagrt::net_queue<std::string> str_vector;
    bool punc_syb = true;
    std::string elem = "";
    for(auto temp : string_value){
        if(temp==cond_syb){
            if(punc_syb)continue;
            else{
                str_vector.emplace_back(elem);
                elem = "";
                punc_syb = true;
            }
        }else {
            punc_syb = false;
            elem.push_back(temp);
        }
    } if(elem.length()) str_vector.emplace_back(elem);
    return str_vector;
}
std::string parse_table(std::string file_path){
    std::ifstream in(file_path);
    if (! in.is_open()){
        std::cout << "Error opening file";
        in.close();
        return " ";
    }else{
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string dat(buffer.str());
        in.close();
        return dat;
    }
}
bagrt::net_queue<std::string> parse_line_strings(std::string strings){
    bagrt::net_queue<std::string> out;
    std::string elem = "";
    for(int i=0; i<strings.length(); ++i){
        char temp = strings.at(i);
        char next_temp = ' ';
        if(i+1 != strings.length()) next_temp = strings.at(i+1);
        else next_temp = ' ';
        if(temp!=' ' && temp!='\t' && temp!='\n' && temp!='\0') elem.push_back(temp);
        else if(next_temp==' ' || next_temp=='\t' || next_temp=='\n' || temp=='\0') continue;
        else{
            out.emplace_back(elem);
            elem = "";
        }
    }out.emplace_back(elem);
    return out;
}
// Output CSV file
template <typename T> void output_table(bagrt::net_queue<bagrt::net_queue<T>>output_strings, std::string file_path){
    std::ofstream oFile;
    oFile.open(file_path,std::ios::out|std::ios::trunc);
    for(auto i=0; i<output_strings.size(); ++i){
        for(auto i=0; i<output_strings[i].size(); ++j) oFile << output_strings[i][j] << ',';
        oFile << std::endl;
    }oFile.close();
}
// Input CSV file
bagrt::net_queue<bagrt::net_queue<std::string>> input_table(std::string file_path){
    auto tab_str = parse_table(file_path);
    auto line_vector = split_string(tab_str, '\n');
    bagrt::net_queue<bagrt::net_queue<std::string>> struct_tab;
    for(auto i=0; i<line_vector.size(); ++i){
        auto tab_vector = split_string(line_vector[i], ',');
        struct_tab.emplace_back(tab_vector);
    }return struct_tab;
}

CSVIO_END