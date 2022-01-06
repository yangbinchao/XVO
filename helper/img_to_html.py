import os
from refile import smart_listdir, smart_isdir, smart_isfile, smart_open, s3_path_join


def gen_html(s3_path_of_image_dir, age_est_type):
    file_names = smart_listdir(s3_path_of_image_dir)
    base_dir_name = os.path.basename(s3_path_of_image_dir)
    html_template = "<!DOCTYPE html><html><head><title>age period</title></head><body><table border='1'><tr><th>Images</th><th>TrueAge / PredAge</th><th>Age--yes/no</th><th>TrueGen / PredGen</th><th>Gender--yes/no</th></tr>{}</table></body></html>"
    # html_template = "<!DOCTYPE html><html><head><title>face_4_TP_FP</title></head><body><table>{}</table></body></html>"
    html_body = ""
    gender_dict = {0:'female', 1:'male'}
    for file_name in file_names:
        info = ""
        file_path = s3_path_join(s3_path_of_image_dir, file_name)
        if smart_isfile(file_path):
            # html_body += "\n"
            # html_body += "<div>img_name: {}</div>".format(file_name) \
            #              + "<div><img src=\"{}\" alt=\"{}\"></div>".format(base_dir_name + "/" + file_name, file_name)
            image = "<div><img src=\"{}\" alt=\"{}\"></div>".format(base_dir_name + "/" + file_name, file_name)
            info += "<td>{}</td>".format(image)
            _, t_age, t_gen, _, p_age, p_gen, _ = file_name.split('_')
            p_gen = p_gen[0]

            Age = t_age + "/" + p_age
            info += "<td>{}</td>".format(Age)

            t_age = float(t_age)
            p_age = float(p_age)

            # 三类容错
            if age_est_type =='3':
                if 0 <= t_age < 18:
                    if abs(t_age - p_age) <= 3:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                elif 18 <= t_age < 65:
                    if abs(t_age - p_age) <= 5:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                else:
                    if abs(t_age - p_age) <= 10:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)

            # 五类容错
            if age_est_type =='5':
                if 0 <= t_age < 6:
                    if 0 <= p_age < 6:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                elif 7 <= t_age < 17:
                    if 7 <= p_age < 17:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                elif 18 <= t_age < 40:
                    if 18 <= p_age < 40:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                elif 41 <= t_age < 65:
                    if 41 <= p_age < 65:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
                elif 65 <= t_age:
                    if 65 <= p_age:
                        Age_Yes_No = 'Yes'
                        info += "<td>{}</td>".format(Age_Yes_No)
                    else:
                        Age_Yes_No = 'No'
                        info += "<td bgcolor='e91e6f'>{}</td>".format(Age_Yes_No)
            
            t_gen, p_gen = int(t_gen), int(p_gen)
            Gender = gender_dict[t_gen] + "/" + gender_dict[p_gen]
            info += "<td>{}</td>".format(Gender)

            if t_gen == p_gen:
                Gender_Yes_No = 'Yes'
                info += "<td>{}</td>".format(Gender_Yes_No)
            else:
                Gender_Yes_No = 'No'
                info += "<td bgcolor='e91e6f'>{}</td>".format(Gender_Yes_No)

            # info = "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>".format(image, Age, Age_Yes_No, Gender, Gender_Yes_No)
            html_body += "<tr>{}</tr>".format(info) 
        else:
            img_names = smart_listdir(file_path)
            html_body += "\n"
            html_body += "<div>img_dir: {}</div>".format(file_name)
            html_body += "<div>"
            for img_name in img_names:
                # html_body += "<span><div>img_name: {}</div>".format(img_name) \
                #              + "<div><img src=\"{}\" alt=\"{}\" width=\"200\"></div></span>" \
                #                  .format(base_dir_name + "/" + file_name + "/" + img_name, img_name)
                html_body += "<span><img src=\"{}\" alt=\"{}\" width=\"200\"></span>".format(
                    base_dir_name + "/" + file_name + "/" + img_name, img_name)
            html_body += "</div>"

    html = html_template.format(html_body)
    return html


def gen_s3_html(s3_path_of_image_dir, oss_site, age_est_type):
    assert smart_isdir(s3_path_of_image_dir)
    assert s3_path_of_image_dir.startswith("s3://")

    if s3_path_of_image_dir.endswith("/"):
        s3_path_of_image_dir = s3_path_of_image_dir[:-1]

    url_path_of_html = "https://oss.iap.{}.brainpp.cn/{}_{}.html".format(oss_site, s3_path_of_image_dir[5:], age_est_type)
    s3_path_of_html = "{}_{}.html".format(s3_path_of_image_dir, age_est_type)

    html = gen_html(s3_path_of_image_dir, age_est_type)

    with smart_open(s3_path_of_html, "w") as f:
        f.write(html)

    return url_path_of_html, s3_path_of_html


def main():

    # s3_path_of_image_dirs = [
    #     "s3://data-wjx/data/algo-mass-production/zyxy_data/21010601003_2021-01-07_zhongyixiongyan/ceshishuju/vehicle/vehicle_data_cropped/test_converted_predicted/tp_fp/vechile_type_tp",
    #     "s3://data-wjx/data/algo-mass-production/zyxy_data/21010601003_2021-01-07_zhongyixiongyan/ceshishuju/vehicle/vehicle_data_cropped/test_converted_predicted/tp_fp/vechile_type_fp",
    #     "s3://data-wjx/data/algo-mass-production/zyxy_data/21010601003_2021-01-07_zhongyixiongyan/ceshishuju/vehicle/vehicle_data_cropped/test_converted_predicted/tp_fp/vechile_color_tp",
    #     "s3://data-wjx/data/algo-mass-production/zyxy_data/21010601003_2021-01-07_zhongyixiongyan/ceshishuju/vehicle/vehicle_data_cropped/test_converted_predicted/tp_fp/vechile_color_fp",
    # ]


    s3_path_of_image_dir = ['s3://data-yrf/data/ZYXY/images/face_3_AAF_val']

    for s3_path_of_image_dir in s3_path_of_image_dir:
        url_path_of_html, s3_path_of_html = gen_s3_html(s3_path_of_image_dir=s3_path_of_image_dir, oss_site="hh-b", age_est_type='5')
        print()
        print("s3_path_of_html = {}".format(s3_path_of_html))
        print("url_path_of_html = {}".format(url_path_of_html))
        print()


if __name__ == '__main__':
    main()

