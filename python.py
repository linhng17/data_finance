import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo State cho Chatbot và API Key ---
# Sử dụng state để lưu trữ lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

# Lấy API Key từ Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")

# Khởi tạo client Gemini (chỉ một lần)
if "client" not in st.session_state:
    if API_KEY:
        try:
            st.session_state.client = genai.Client(api_key=API_KEY)
        except Exception as e:
            st.error(f"Lỗi khởi tạo Gemini Client: {e}")
            st.session_state.client = None
    else:
        # Nếu API Key không tồn tại, client sẽ là None
        st.session_state.client = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 thủ công cho mẫu số để tính tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Hàm cũ) ---
def get_ai_analysis(data_for_ai, client):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not client:
        return "Lỗi gọi Gemini API: Client chưa được khởi tạo do thiếu Khóa API."
        
    try:
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"
        
# --- Hàm khởi tạo Chat Session ---
def initialize_chat_session(df_processed_markdown):
    """Khởi tạo Chat Session với System Instruction."""
    client = st.session_state.get("client")
    
    # BỔ SUNG KIỂM TRA: Nếu client không tồn tại, báo lỗi ngay lập tức
    if not client:
        st.error("Lỗi khởi tạo Chatbot: Không tìm thấy Gemini Client (thiếu Khóa API).", icon="🚨")
        return False
    
    # System instruction để Gemini hiểu ngữ cảnh
    system_instruction = f"""
    Bạn là một Trợ lý phân tích tài chính chuyên nghiệp và thân thiện, có khả năng trả lời các câu hỏi dựa trên dữ liệu Báo cáo Tài chính được cung cấp.
    
    Dữ liệu phân tích tài chính hiện tại của công ty:
    {df_processed_markdown}
    
    Hãy sử dụng dữ liệu này để trả lời các câu hỏi của người dùng. Nếu thông tin không có trong bảng, hãy trả lời theo kiến thức tài chính chung. Luôn trả lời bằng Tiếng Việt.
    """
    
    try:
        # Sử dụng genai.types.GenerateContentConfig để truyền system_instruction (đã sửa lỗi TypeError trước đó)
        st.session_state.chat_session = client.chats.create(
            model="gemini-2.5-flash",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        st.session_state.chat_history.append({"role": "model", "content": "Chào bạn! Tôi là Trợ lý AI. Hãy hỏi tôi về dữ liệu tài chính bạn vừa tải lên nhé."})
        return True
    except APIError as e:
        st.error(f"Lỗi khởi tạo Chat Session: Lỗi API ({e}). Vui lòng kiểm tra Khóa API.", icon="🚨")
        return False
    except Exception as e:
        st.error(f"Lỗi khởi tạo Chat Session: Đã xảy ra lỗi không xác định ({e}).", icon="🚨")
        return False


# --- Hàm xử lý Chatbot cho Pop-up ---
def handle_chatbot_input_popup(user_prompt, chat_container):
    """Xử lý đầu vào từ người dùng và gửi đến chat session của Gemini."""
    
    if st.session_state.chat_session is None:
        chat_container.error("Chatbot chưa sẵn sàng. Vui lòng nhấn nút **Bật/Reset Chat**.", icon="🚨")
        return

    # Thêm câu hỏi người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Gửi tin nhắn đến Gemini và hiển thị kết quả
    try:
        # Hiển thị tin nhắn người dùng
        # Sử dụng chat_container để hiển thị tin nhắn trong hộp lịch sử
        with chat_container.chat_message("user"):
            st.markdown(user_prompt)

        # Hiển thị phản hồi của model
        with chat_container.chat_message("model"):
            with st.spinner("Đang phân tích..."):
                # SỬA LỖI: Đoạn code này bị lỗi timeout hoặc lỗi API call
                response = st.session_state.chat_session.send_message(user_prompt)
                model_response = response.text
            
            st.markdown(model_response)
        
        # Thêm câu trả lời của model vào lịch sử
        st.session_state.chat_history.append({"role": "model", "content": model_response})
        
    except APIError as e:
        # SỬA LỖI: Cập nhật thông báo lỗi chi tiết hơn
        error_message = f"Lỗi API: Không thể nhận phản hồi. Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết: {e}"
        st.error(error_message, icon="🚨")
        st.session_state.chat_history.append({"role": "model", "content": "Lỗi: Không thể nhận phản hồi từ AI do lỗi API."})
    except Exception as e:
        error_message = f"Lỗi không xác định: {e}"
        st.error(error_message, icon="🚨")
        st.session_state.chat_history.append({"role": "model", "content": "Lỗi: Đã xảy ra lỗi bất ngờ."})


# Khởi tạo biến để giữ dữ liệu đã xử lý
df_processed = None
df_processed_markdown = "" # Chuỗi markdown của dữ liệu đã xử lý

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            df_processed_markdown = df_processed.to_markdown(index=False) # Tạo markdown cho Chatbot context
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định cho Chức năng 5
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tránh chia cho 0
                divisor_no_n = no_ngan_han_N if no_ngan_han_N != 0 else 1e-9
                divisor_no_n_1 = no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 1e-9
                
                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / divisor_no_n
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / divisor_no_n_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                st.warning("Không thể tính Chỉ số Thanh toán Hiện hành do Nợ Ngắn Hạn bằng 0.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                
            # --- Chức năng 5: Nhận xét AI Tự động ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            tsnh_tg_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
            tsnh_tg = f"{tsnh_tg_row['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not tsnh_tg_row.empty else "N/A"
            
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed_markdown,
                    tsnh_tg, 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                client = st.session_state.get("client")
                
                if client:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, client)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Lỗi: Không thể thực hiện Phân tích AI do thiếu Khóa API.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và các cột.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    

# ******************************************************************************
# --- PHẦN BỔ SUNG KHUNG CHAT RIÊNG BIỆT (Pop-up mô phỏng) ---
# ******************************************************************************
st.markdown("---")
st.subheader("6. Trợ lý Chatbot AI")
st.markdown("*(Dùng để hỏi đáp chuyên sâu về dữ liệu tài chính bạn đã tải lên)*")

# Nút Bật/Reset Chat
if df_processed is not None:
    # Kiểm tra lại client trước khi cố gắng khởi tạo chat
    if st.button("Bật/Reset Chat", key="reset_chat", type="primary"):
        st.session_state.chat_session = None # Xóa session cũ
        st.session_state.chat_history = [] # Xóa lịch sử
        
        # Khởi tạo session mới sau khi reset
        client = st.session_state.get("client")
        if client and df_processed_markdown:
            if initialize_chat_session(df_processed_markdown):
                # Khởi tạo thành công, chỉ cần rerender
                st.rerun() 
            # initialize_chat_session sẽ tự hiển thị lỗi nếu có
        else:
            st.error("Lỗi: Không tìm thấy Khóa API hoặc Client chưa sẵn sàng. Vui lòng kiểm tra cấu hình Khóa API.", icon="🚨")
else:
    st.info("Tải file lên trước để kích hoạt chatbot.")


# --- Khu vực Chat Interface ---
if st.session_state.chat_session is not None:
    
    # Container để chứa lịch sử chat
    chat_history_container = st.container(height=400, border=True)

    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with chat_history_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input cho người dùng
    user_prompt = st.chat_input("Hỏi AI về các chỉ số tài chính...", key="chat_input_main")
    
    if user_prompt:
        # Nếu người dùng nhập, gọi hàm xử lý chat
        # SỬA LỖI: Cần truyền chat_history_container vào hàm xử lý
        handle_chatbot_input_popup(user_prompt, chat_history_container)
        st.rerun() # Tải lại trang để hiển thị tin nhắn mới ngay lập tức
else:
    st.info("Nhấn nút **'Bật/Reset Chat'** để khởi động phiên trò chuyện dựa trên dữ liệu tài chính của bạn.")
