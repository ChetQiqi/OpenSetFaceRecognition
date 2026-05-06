#!/usr/bin/env python3
import sys
from pathlib import Path
import os

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.chdir(str(project_root))

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from collections import defaultdict

# 设置页面配置
st.set_page_config(
    page_title="人脸识别系统",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """初始化 session state"""
    if 'db_path' not in st.session_state:
        # 使用实际的数据库路径
        db_path = project_root / "benchmark" / "YTF_100p.db"
        st.session_state.db_path = str(db_path.absolute())

    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'gallery' not in st.session_state:
        st.session_state.gallery = None


@st.cache_resource
def load_model(db_path: str):
    """加载模型和特征库（带缓存，db_path 作为缓存 key）"""
    try:
        from apps.recognition_system.core.operations import build_runtime
        from apps.recognition_system.core.feature_db import FeatureDB

        weights_path = project_root / "weights" / "model_best.pt"

        # 加载模型
        model, detector = build_runtime(
            weights_path=str(weights_path.absolute()),
            model_name="iresnet50",
            img_size=112,
            device="auto",
            det_conf_threshold=0.60,
            det_min_size=40,
            detector_backend="mtcnn"
        )

        # 加载特征库（临时 db，只用于加载 gallery，不缓存连接）
        with FeatureDB(db_path) as db:
            gallery = db.load_gallery(mode="mean")

        return model, detector, gallery

    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        import traceback
        st.error(f"详细错误:\n{traceback.format_exc()}")
        return None, None, None


def get_db():
    """每次调用都创建新的数据库连接（线程安全）"""
    from apps.recognition_system.core.feature_db import FeatureDB
    return FeatureDB(st.session_state.db_path)


def get_stats():
    """获取统计信息"""
    try:
        with get_db() as db:
            stats = db.get_stats()
        return stats.get("person_count", 0), stats.get("embedding_count", 0)
    except Exception as e:
        st.warning(f"获取统计信息失败: {e}")
        return 0, 0


def show_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.markdown("## 🎛️ 控制面板")

        # 数据库信息
        st.markdown("### 📊 系统状态")

        if st.session_state.model_loaded:
            st.success("✅ 模型已加载")

            # 获取统计信息（每次创建新连接，线程安全）
            total_persons, total_features = get_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("注册人数", total_persons, delta=None)
            with col2:
                st.metric("特征数量", total_features, delta=None)
        else:
            st.warning("⚠️ 模型未加载")
            if st.button("🚀 加载模型", width="stretch"):
                with st.spinner("正在加载模型..."):
                    model, detector, gallery = load_model(st.session_state.db_path)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.detector = detector
                        st.session_state.gallery = gallery
                        st.session_state.model_loaded = True
                        st.rerun()

        st.markdown("---")

        # 配置
        st.markdown("### ⚙️ 配置")
        new_db_path = st.text_input("📁 数据库路径", st.session_state.db_path)
        if new_db_path != st.session_state.db_path:
            st.session_state.db_path = new_db_path
            st.session_state.model_loaded = False
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # 关于信息
        st.markdown("### ℹ️ 关于")
        st.info("""
        **人脸识别系统 v3.0**

        - 🎯 MTCNN 人脸检测
        - 🧠 iResNet50 特征提取
        - 💾 SQLite 特征存储
        - 🌐 Streamlit Web UI

        © 2026 Face Recognition System
        """)


def page_home():
    """主页"""
    # 大标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin-bottom: 0.5rem;">
            👤 人脸识别系统
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: 0;">
            基于深度学习的智能人脸识别平台
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 系统状态概览
    if st.session_state.model_loaded:
        total_persons, total_features = get_stats()

        st.markdown("### 📊 系统概览")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 2rem;">{total_persons}</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">注册人员</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 2rem;">{total_features}</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">特征数量</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 2rem;">✓</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">模型已就绪</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 2rem;">🚀</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">系统在线</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 请先在侧边栏加载模型以开始使用")

    st.markdown("<br>", unsafe_allow_html=True)

    # 功能模块卡片
    st.markdown("### 🎯 核心功能")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background: white; border: 2px solid #667eea; border-radius: 15px;
                    padding: 2rem; text-align: center; height: 200px;
                    transition: transform 0.3s; cursor: pointer;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👥</div>
            <h3 style="color: #667eea; margin: 0.5rem 0;">人员管理</h3>
            <p style="color: #666; font-size: 0.9rem;">注册与管理人员信息</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; border: 2px solid #f093fb; border-radius: 15px;
                    padding: 2rem; text-align: center; height: 200px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
            <h3 style="color: #f093fb; margin: 0.5rem 0;">实时识别</h3>
            <p style="color: #666; font-size: 0.9rem;">摄像头实时人脸识别</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: white; border: 2px solid #4facfe; border-radius: 15px;
                    padding: 2rem; text-align: center; height: 200px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🖼️</div>
            <h3 style="color: #4facfe; margin: 0.5rem 0;">图片识别</h3>
            <p style="color: #666; font-size: 0.9rem;">上传图片批量识别</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: white; border: 2px solid #43e97b; border-radius: 15px;
                    padding: 2rem; text-align: center; height: 200px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎬</div>
            <h3 style="color: #43e97b; margin: 0.5rem 0;">视频识别</h3>
            <p style="color: #666; font-size: 0.9rem;">视频文件人员追踪</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 快速开始指南
    col_guide1, col_guide2 = st.columns(2)

    with col_guide1:
        st.markdown("### 🚀 快速开始")
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
            <h4 style="color: #667eea; margin-top: 0;">使用步骤</h4>
            <ol style="color: #333; line-height: 1.8;">
                <li><strong>加载模型</strong> - 点击侧边栏"加载模型"按钮</li>
                <li><strong>注册人员</strong> - 在"人员管理"页面注册新人员</li>
                <li><strong>开始识别</strong> - 选择识别模式进行人脸识别</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_guide2:
        st.markdown("### ⚡ 技术特性")
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #f5576c;">
            <h4 style="color: #f5576c; margin-top: 0;">核心技术</h4>
            <ul style="color: #333; line-height: 1.8; list-style: none; padding-left: 0;">
                <li>✨ <strong>MTCNN</strong> 人脸检测算法</li>
                <li>🧠 <strong>iResNet50</strong> 特征提取网络</li>
                <li>💾 <strong>SQLite</strong> 特征向量存储</li>
                <li>📷 <strong>OpenCV</strong> 实时摄像头识别</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 底部提示
    st.info("💡 **提示**: 使用左侧导航栏切换不同功能模块，开始您的人脸识别之旅！")


def page_person_management():
    """人员管理页面"""
    st.markdown("## 👥 人员管理")

    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先在侧边栏加载模型！")
        return

    model, detector, gallery = load_model(st.session_state.db_path)

    tab1, tab2, tab3, tab4 = st.tabs(["➕ 注册新人", "📋 人员列表", "🗑️ 删除人员", "✏️ 编辑人员"])

    with tab1:
        st.markdown("### 注册新人员")

        person_id = st.text_input("👤 人员 ID", placeholder="例如: Alice")

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "📸 上传人脸照片（可多选）",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True
            )

        with col2:
            st.markdown("**要求：**")
            st.markdown("- 清晰的正面照")
            st.markdown("- 光线充足")
            st.markdown("- 无遮挡")
            st.markdown("- 建议 3-5 张")

        if st.button("✅ 注册", type="primary", width="stretch"):
            if not person_id:
                st.error("❌ 请输入人员 ID！")
            elif not uploaded_files:
                st.error("❌ 请上传至少一张照片！")
            else:
                # 处理注册
                progress_bar = st.progress(0)
                status_text = st.empty()

                success_count = 0
                fail_count = 0
                from apps.recognition_system.core.operations import extract_face_embedding

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"处理中: {idx + 1}/{len(uploaded_files)}")

                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    try:
                        feature = extract_face_embedding(image, model, detector)
                        if feature is None:
                            fail_count += 1
                            st.warning(f"⚠️ {uploaded_file.name} - 未检测到人脸")
                        else:
                            with get_db() as db:
                                db.add_embedding(person_id, feature)
                            success_count += 1
                    except Exception as e:
                        fail_count += 1
                        st.warning(f"⚠️ {uploaded_file.name} - 处理失败: {e}")

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.empty()

                if success_count > 0:
                    st.success(f"✅ 成功注册 {success_count} 张照片！")
                    if fail_count > 0:
                        st.warning(f"⚠️ {fail_count} 张照片处理失败")
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 注册失败！请检查照片质量")
 
    with tab2:
        st.markdown("### 已注册人员")
        try:
            with get_db() as db:
                persons_list = db.list_persons()
            if not persons_list:
                st.info("📭 暂无注册人员")
            else:
                # 添加搜索框
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_query = st.text_input(
                        "🔍 搜索人员",
                        placeholder="输入人员 ID 或姓名...",
                        key="person_search"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 刷新列表"):
                        st.rerun()

                # 根据搜索条件过滤人员列表
                if search_query:
                    filtered_persons = [
                        (name, count) for name, count in persons_list
                        if search_query.lower() in name.lower()
                    ]
                else:
                    filtered_persons = persons_list

                # 显示统计信息
                if search_query:
                    st.markdown(f"**找到 {len(filtered_persons)} 人（共 {len(persons_list)} 人）**")
                else:
                    st.markdown(f"**共 {len(persons_list)} 人**")

                # 显示人员列表
                if not filtered_persons:
                    st.warning(f"⚠️ 未找到包含 '{search_query}' 的人员")
                else:
                    # 使用列布局显示，更美观
                    for idx, (person_name, feature_count) in enumerate(filtered_persons):
                        with st.container():
                            col_icon, col_info = st.columns([0.5, 9.5])
                            with col_icon:
                                st.markdown("👤")
                            with col_info:
                                st.markdown(f"**{person_name}** - {feature_count} 张特征")
                            if idx < len(filtered_persons) - 1:
                                st.divider()
        except Exception as e:
            st.error(f"❌ 加载人员列表失败: {e}")

    with tab3:
        st.markdown("### 删除人员")
        try:
            with get_db() as db:
                persons_list = db.list_persons()
            if not persons_list:
                st.info("📭 暂无注册人员")
            else:
                person_names = [name for name, count in persons_list]
                person_to_delete = st.selectbox("选择要删除的人员", person_names)

                st.warning(f"⚠️ 确定要删除 **{person_to_delete}** 吗？此操作不可恢复！")

                if st.button("🗑️ 确认删除", type="primary"):
                    try:
                        with get_db() as db:
                            db.delete_person(person_to_delete)
                        st.success(f"✅ 已删除 {person_to_delete}")
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败: {e}")
        except Exception as e:
            st.error(f"❌ 加载人员列表失败: {e}")

    with tab4:
        st.markdown("### 编辑人员信息")
        try:
            with get_db() as db:
                persons_list = db.list_persons()
            if not persons_list:
                st.info("📭 暂无注册人员")
            else:
                person_names = [name for name, count in persons_list]

                col1, col2 = st.columns(2)
                with col1:
                    person_to_edit = st.selectbox("🔍 选择要编辑的人员", person_names, key="person_to_edit")

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 刷新列表", key="refresh_edit"):
                        st.rerun()

                st.divider()

                # 显示当前人员信息
                current_person = person_to_edit
                for name, count in persons_list:
                    if name == current_person:
                        st.markdown(f"**当前人员:** {current_person}  |  **特征数:** {count}")
                        break

                st.markdown("#### 修改人员名字")
                new_person_name = st.text_input(
                    "👤 新名字",
                    value=current_person,
                    placeholder="输入新的人员名字...",
                    key="new_person_name"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 保存修改", type="primary", use_container_width=True):
                        if not new_person_name:
                            st.error("❌ 新名字不能为空！")
                        elif new_person_name == current_person:
                            st.warning("⚠️ 新名字与原名字相同，无需修改")
                        else:
                            try:
                                with get_db() as db:
                                    db.rename_person(current_person, new_person_name)
                                st.success(f"✅ 已将 '{current_person}' 修改为 '{new_person_name}'")
                                st.cache_resource.clear()
                                time.sleep(1)
                                st.rerun()
                            except ValueError as e:
                                st.error(f"❌ {str(e)}")
                            except Exception as e:
                                st.error(f"❌ 修改失败: {e}")

                with col2:
                    if st.button("❌ 取消", use_container_width=True):
                        st.rerun()

        except Exception as e:
            st.error(f"❌ 加载人员列表失败: {e}")


def page_image_recognition():
    """图片识别页面"""
    st.markdown("## 🖼️ 图片识别")

    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先在侧边栏加载模型！")
        return

    model, detector, gallery = load_model(st.session_state.db_path)

    uploaded_file = st.file_uploader(
        "📸 上传图片",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file is not None:
        # 读取图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 显示原图
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📷 原始图片")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, width="stretch")

        with st.spinner("正在识别..."):
            try:
                from apps.recognition_system.core.operations import recognize_faces, draw_recognitions

                results = recognize_faces(
                    image, model, detector, gallery,
                    threshold=0.45, match_reduce="topk_mean", topk=3
                )

                if not results:
                    with col2:
                        st.markdown("### 🎯 识别结果")
                        st.warning("⚠️ 未检测到人脸")
                else:
                    annotated_image = draw_recognitions(image, results)

                    with col2:
                        st.markdown("### 🎯 识别结果")
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image_rgb, width="stretch")

                    st.markdown("---")
                    st.markdown("### 📊 识别详情")

                    # 统计已知人员和陌生人
                    known_count = sum(1 for r in results if r['accepted'])
                    stranger_count = sum(1 for r in results if not r['accepted'])

                    st.success(f"✅ 检测到 {len(results)} 个人脸 (已知: {known_count}, 陌生人: {stranger_count})")

                    for idx, result in enumerate(results):
                        # 使用 display_name 显示中文标签
                        display_name = result.get('display_name', result['name'])
                        status_icon = "✅" if result['accepted'] else "⚠️"

                        with st.expander(f"{status_icon} 人物 {idx + 1}: {display_name}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("置信度", f"{result['score'] * 100:.1f}%")
                            with col_b:
                                box = result['box']
                                st.text(f"位置: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

                            # 置信度进度条（陌生人用红色提示）
                            confidence = result['score']
                            if result['accepted']:
                                st.progress(confidence)
                            else:
                                st.progress(confidence)
                                st.warning("⚠️ 相似度低于阈值，标记为陌生人")

                    # 保存结果
                    if st.button("💾 保存标注图片"):
                        output_path = Path(tempfile.gettempdir()) / f"recognized_{uploaded_file.name}"
                        cv2.imwrite(str(output_path), annotated_image)
                        st.success(f"✅ 已保存到: {output_path}")

            except Exception as e:
                with col2:
                    st.markdown("### 🎯 识别结果")
                    st.error(f"❌ 识别失败: {e}")
                import traceback
                st.error(f"详细错误:\n{traceback.format_exc()}")


def page_video_recognition():
    """视频识别页面 - Web中显示识别效果"""
    st.markdown("## 🎬 视频识别")

    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先在侧边栏加载模型！")
        return

    model, detector, gallery = load_model(st.session_state.db_path)

    # 功能模式选择
    st.markdown("### 🎯 功能模式")
    mode = st.radio(
        "选择功能",
        options=["视频中找人", "验证ID"],
        horizontal=True,
        help="视频中找人：识别视频中所有人员 | 验证ID：验证指定人员是否出现"
    )

    # 如果是验证ID模式，需要选择要验证的ID
    verify_target = None
    if mode == "验证ID":
        # 获取所有已注册的人员
        from apps.recognition_system.core.feature_db import FeatureDB
        try:
            with FeatureDB(st.session_state.db_path) as db:
                persons_list = db.list_persons()
        except Exception as e:
            st.error(f"❌ 无法读取特征库: {e}")
            return

        if not persons_list:
            st.error("❌ 特征库为空，请先在侧边栏注册人员！")
            return

        person_names = [name for name, count in persons_list]
        verify_target = st.selectbox(
            "🆔 选择要验证的ID",
            options=person_names,
            help="选择需要在视频中验证的人员ID"
        )
        st.info(f"📋 将在视频中验证：**{verify_target}**")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "🎥 上传视频",
        type=['mp4', 'avi', 'mov', 'mkv']
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        skip_frames = st.slider("跳帧数（越大越快）", 1, 10, 5,
                               help="每隔N帧识别一次，越小越精确")
    with col2:
        threshold = st.slider("识别阈值", 0.0, 1.0, 0.45, 0.05,
                             help="相似度阈值，越高越严格")
    with col3:
        stable_frames = st.slider("稳定帧数", 2, 10, 3,
                                 help="连续N帧识别为同一人才显示，避免误识别")

    if uploaded_file is not None:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.success(f"✅ 视频已上传: {uploaded_file.name}")

        if st.button("🚀 开始识别处理", type="primary", width="stretch"):
            try:
                from apps.recognition_system.core.operations import recognize_faces, draw_recognitions
                from apps.recognition_system.core.tracker import FaceTracker
                import time

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.info(f"📹 视频信息: {total_frames} 帧，{fps} FPS，{width}x{height}")

                # 输出视频路径（先输出为临时文件，再转换）
                temp_output = Path(tempfile.gettempdir()) / f"temp_{int(time.time())}.mp4"
                output_path = Path(tempfile.gettempdir()) / f"recognized_{int(time.time())}_{uploaded_file.name}"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

                # 创建人脸跟踪器（时序平滑）
                tracker = FaceTracker(
                    history_size=stable_frames,
                    min_stable_count=stable_frames,
                    iou_threshold=0.3
                )

                # 进度条
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 统计
                person_counts = {}
                stranger_count = 0  # 陌生人计数
                frame_idx = 0
                processed_frames = 0
                start_time = time.time()
                last_results = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 跳帧处理
                    if frame_idx % skip_frames == 0:
                        try:
                            # 原始识别结果
                            raw_results = recognize_faces(
                                frame, model, detector, gallery,
                                threshold=threshold, match_reduce="topk_mean", topk=3
                            )

                            # 使用跟踪器进行时序平滑
                            stable_results = tracker.update(raw_results)
                            last_results = stable_results

                            # 统计稳定后的结果
                            if stable_results:
                                for r in stable_results:
                                    if r["accepted"]:
                                        name = r["name"]
                                        person_counts[name] = person_counts.get(name, 0) + 1
                                    else:
                                        # 统计陌生人
                                        stranger_count += 1
                            processed_frames += 1
                        except Exception as e:
                            # 静默处理错误，避免刷屏
                            last_results = []

                    # 绘制识别结果
                    display_frame = frame.copy()
                    if last_results:
                        display_frame = draw_recognitions(display_frame, last_results)

                    # 写入输出视频
                    out.write(display_frame)
                    frame_idx += 1

                    # 更新进度
                    progress = frame_idx / total_frames
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    status_text.text(f"处理中: {frame_idx}/{total_frames} 帧 ({progress*100:.1f}%) - 速度: {current_fps:.1f} FPS")

                cap.release()
                out.release()

                # 使用 ffmpeg 转换为浏览器兼容的 H.264 编码
                status_text.text("🔄 正在转换视频编码...")
                try:
                    import subprocess
                    # 使用 ffmpeg 转换为 H.264 编码
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',  # 覆盖输出文件
                        '-i', str(temp_output),  # 输入文件
                        '-c:v', 'libx264',  # H.264 编码
                        '-preset', 'fast',  # 编码速度
                        '-crf', '23',  # 质量（18-28，越小质量越好）
                        '-c:a', 'aac',  # 音频编码
                        '-movflags', '+faststart',  # 优化网络播放
                        str(output_path)  # 输出文件
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

                    if result.returncode != 0:
                        # ffmpeg 失败，使用原始视频
                        st.warning("⚠️ 视频转码失败，使用原始编码（可能无法在浏览器中播放）")
                        import shutil
                        shutil.copy(str(temp_output), str(output_path))

                    # 删除临时文件
                    temp_output.unlink(missing_ok=True)

                except Exception as e:
                    st.warning(f"⚠️ 视频转码出错: {e}，使用原始编码")
                    # 如果转换失败，复制原视频
                    import shutil
                    if temp_output.exists():
                        shutil.copy(str(temp_output), str(output_path))
                        temp_output.unlink(missing_ok=True)

                # 显示结果
                status_text.empty()
                progress_bar.empty()
                st.success("✅ 处理完成！")

                st.markdown("---")

                # 显示统计信息
                st.markdown("### 📊 识别统计")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总帧数", total_frames)
                with col2:
                    st.metric("处理帧数", frame_idx)
                with col3:
                    st.metric("识别帧数", processed_frames)
                with col4:
                    st.metric("识别人数", len(person_counts))

                # 显示识别到的人员
                if mode == "视频中找人":
                    # 模式1: 显示所有识别到的人员
                    if person_counts or stranger_count > 0:
                        st.markdown("### 👥 识别到的人员")

                        # 已知人员
                        if person_counts:
                            person_names = sorted(person_counts.keys())
                            st.markdown("**识别到的人员：**")

                            # 每行显示5个人的卡片
                            cols = st.columns(5)
                            for idx, name in enumerate(person_names):
                                with cols[idx % 5]:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                padding: 0.8rem; border-radius: 10px;
                                                text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <p style="color: white; margin: 0; font-size: 1rem;">
                                            👤 {name}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # 陌生人提示（只有比例 > 5% 时才显示）
                        if processed_frames > 0:
                            stranger_ratio = stranger_count / processed_frames
                            if stranger_ratio > 0.05:  # 5% 的阈值
                                st.info(f"⚠️ 视频中已检测到陌生人 ({stranger_ratio*100:.1f}%)")
                    else:
                        st.info("ℹ️ 视频中未识别到任何人员")

                else:  # mode == "验证ID"
                    # 模式2: 验证指定ID
                    st.markdown("### 🎯 验证结果")

                    if verify_target in person_counts:
                        verify_frames = person_counts[verify_target]
                        # 验证通过：至少10帧
                        if verify_frames >= 10:
                            st.success(f"✅ **{verify_target}** 验证通过 😊")
                        else:
                            st.warning(f"⚠️ **{verify_target}** 检测到但次数不足 ({verify_frames} 帧 < 10 帧)")
                    else:
                        st.error(f"❌ **{verify_target}** 未找到")

                st.markdown("---")

                # 在Web中直接显示识别后的视频（固定尺寸窗口）
                st.markdown("### 🎥 识别结果视频")
                if output_path.exists():
                    # 读取视频文件并转换为base64用于HTML嵌入
                    import base64
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        video_base64 = base64.b64encode(video_bytes).decode()

                    # 使用HTML video标签，固定窗口大小 800x600px
                    video_html = f"""
                    <div style="display: flex; justify-content: center; margin: 20px 0;">
                        <video
                            width="800"
                            height="600"
                            controls
                            style="
                                width: 800px;
                                height: 600px;
                                object-fit: contain;
                                background-color: #000;
                                border: 2px solid #ddd;
                                border-radius: 10px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            ">
                            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                            您的浏览器不支持视频播放。
                        </video>
                    </div>
                    """
                    st.markdown(video_html, unsafe_allow_html=True)

                    # 提供下载按钮
                    st.markdown("")  # 空行
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        st.download_button(
                            label="📥 下载识别后的视频",
                            data=video_bytes,
                            file_name=f"recognized_{uploaded_file.name}",
                            mime="video/mp4",
                            width="stretch"
                        )

                    st.info(f"💾 视频已保存至: {output_path}")
                else:
                    st.error("❌ 输出视频文件不存在")

            except Exception as e:
                st.error(f"❌ 处理失败: {e}")
                import traceback
                st.error(f"详细错误:\n{traceback.format_exc()}")
            finally:
                # 清理上传的临时文件（保留输出文件供用户查看）
                try:
                    Path(video_path).unlink(missing_ok=True)
                except:
                    pass


def page_camera():
    """摄像头实时识别页面 - 使用线程化OpenCV"""
    st.markdown("## 📷 摄像头实时识别")

    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先在侧边栏加载模型！")
        return

    # 功能模式选择
    st.markdown("### 🎯 功能模式")
    mode = st.radio(
        "选择功能",
        options=["视频中找人", "验证ID"],
        horizontal=True,
        help="视频中找人：识别所有出现的人员 | 验证ID：验证指定人员是否出现",
        key="camera_mode"
    )

    # 如果是验证ID模式，需要选择要验证的ID
    verify_target = None
    if mode == "验证ID":
        # 获取所有已注册的人员
        from apps.recognition_system.core.feature_db import FeatureDB
        try:
            with FeatureDB(st.session_state.db_path) as db:
                persons_list = db.list_persons()
        except Exception as e:
            st.error(f"❌ 无法读取特征库: {e}")
            return

        if not persons_list:
            st.error("❌ 特征库为空，请先在侧边栏注册人员！")
            return

        person_names = [name for name, count in persons_list]
        verify_target = st.selectbox(
            "🆔 选择要验证的ID",
            options=person_names,
            help="选择需要验证的人员ID",
            key="camera_verify_target"
        )
        st.info(f"📋 待验证人员：**{verify_target}**")

    st.markdown("---")

    # 参数控制
    st.markdown("### ⚙️ 识别参数")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        skip_frames = st.slider("跳帧数", 1, 10, 3,
                                help="每隔N帧识别一次，越小越精确",
                                key="camera_skip_frames")
    with col2:
        threshold = st.slider("识别阈值", 0.0, 1.0, 0.45, 0.05,
                             help="相似度阈值，越高越严格",
                             key="camera_threshold")
    with col3:
        stable_frames = st.slider("稳定帧数", 2, 10, 3,
                                 help="连续N帧识别为同一人才显示",
                                 key="camera_stable_frames")
    with col4:
        camera_id = st.number_input("摄像头ID", 0, 10, 0,
                                   help="通常是0，多摄像头时可调整",
                                   key="camera_id")

    st.markdown("---")

    # 初始化摄像头控制状态
    if "camera_thread" not in st.session_state:
        st.session_state.camera_thread = None
        st.session_state.camera_running = False
        st.session_state.camera_data = {
            "results": [],
            "stats": defaultdict(int),
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0.0,
            "running": False,
            "verify_status": None,  # 验证状态: success/failure/None
            "verify_count": 0,      # 验证目标出现次数
        }

    # 检查线程是否还活着（用户可能按了 Q 键关闭弹窗）
    if st.session_state.camera_running:
        t = st.session_state.camera_thread
        if t is None or not t.is_alive():
            st.session_state.camera_running = False
            st.session_state.camera_thread = None

    col_start, col_stop, _ = st.columns([1, 1, 2])
    with col_start:
        if st.button("🟢 启动摄像头", disabled=st.session_state.camera_running,
                    width="stretch", type="primary"):
            from apps.recognition_system.core.camera_thread import CameraThread

            # 重置统计数据（每次启动清空）
            st.session_state.camera_data = {
                "results": [],
                "stats": defaultdict(int),
                "total_frames": 0,
                "processed_frames": 0,
                "fps": 0.0,
                "running": True,
                "verify_status": None,
                "verify_count": 0,
            }

            thread = CameraThread(
                camera_id=camera_id,
                skip_frames=skip_frames,
                threshold=threshold,
                model=st.session_state.model,
                detector=st.session_state.detector,
                gallery=st.session_state.gallery,
                data_dict=st.session_state.camera_data,
                stable_frames=stable_frames,
                mode=mode,
                verify_target=verify_target
            )
            thread.start()
            st.session_state.camera_thread = thread
            st.session_state.camera_running = True
            st.rerun()

    with col_stop:
        if st.button("🔴 停止摄像头", disabled=not st.session_state.camera_running,
                    width="stretch"):
            if st.session_state.camera_thread:
                st.session_state.camera_thread.stop()
                st.session_state.camera_thread = None
                st.session_state.camera_running = False
                # 不清空camera_data，保留结果供查看
            st.rerun()

    st.markdown("---")

    if st.session_state.camera_running:
        st.success("🟢 摄像头运行中 — 识别画面在弹出的独立窗口显示，按 **Q** 可关闭")

        data = st.session_state.camera_data
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("总帧数", data["total_frames"])
        col_meta2.metric("已识别帧", data["processed_frames"])
        col_meta3.metric("FPS", f"{data['fps']:.1f}")

        st.markdown("---")
        col_detect, col_stats = st.columns(2)

        with col_detect:
            st.markdown("#### 🎯 当前检测")
            results = data.get("results", [])
            if results:
                for r in results:
                    display_name = r.get('display_name', r.get('name', 'Unknown'))
                    if r.get("accepted", False):
                        st.success(f"✅ **{display_name}** ({r['score']*100:.1f}%)")
                    else:
                        st.error(f"⚠️ **{display_name}** ({r['score']*100:.1f}%)")
            else:
                st.info("等待检测人脸...")

        with col_stats:
            if mode == "视频中找人":
                # 模式1: 显示所有识别到的人员
                st.markdown("#### 📈 已识别人员")
                stats = data.get("stats", {})
                if stats:
                    # 将 "Unknown" 标记为陌生人，其他为已知人员
                    known_persons = [name for name in stats.keys() if name != "Unknown"]
                    has_strangers = "Unknown" in stats and stats.get("Unknown", 0) >= 5  # 至少5次陌生人检测

                    if known_persons:
                        st.write("**已识别人员:**")
                        cols = st.columns(5)
                        for idx, name in enumerate(sorted(known_persons)):
                            with cols[idx % 5]:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            padding: 0.8rem; border-radius: 10px;
                                            text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                    <p style="color: white; margin: 0; font-size: 1rem;">
                                        👤 {name}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                    if has_strangers:
                        stranger_count = stats.get("Unknown", 0)
                        st.info(f"⚠️ 已检测到陌生人 ({stranger_count} 次)")
                else:
                    st.info("暂无识别记录")

            else:  # mode == "验证ID"
                # 模式2: 验证指定ID
                st.markdown("#### 🎯 验证结果")
                stats = data.get("stats", {})

                if verify_target in stats:
                    verify_count = stats[verify_target]
                    # 验证通过：至少5次检测
                    if verify_count >= 5:
                        st.success(f"✅ **{verify_target}** 验证通过 😊")
                    else:
                        st.warning(f"⚠️ **{verify_target}** 检测到但次数不足 ({verify_count} 次 < 5 次)")
                else:
                    st.info(f"等待检测 {verify_target}...")

        # 自动刷新
        time.sleep(0.5)
        st.rerun()

    else:
        # 摄像头未运行
        data = st.session_state.camera_data

        # 验证ID模式：显示最终验证结果
        if mode == "验证ID" and data.get("verify_status"):
            st.markdown("---")
            st.markdown("### 🎯 验证结果")

            if data["verify_status"] == "success":
                st.success(f"✅ **{verify_target}** 验证通过 😊")
            elif data["verify_status"] == "failure":
                st.error(f"❌ **{verify_target}** 验证失败 ({data.get('verify_count', 0)} 次 < 5 次，已处理 {data.get('processed_frames', 0)} 帧)")

            # 显示统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总帧数", data.get("total_frames", 0))
            with col2:
                st.metric("已识别帧", data.get("processed_frames", 0))
            with col3:
                st.metric("目标出现", f"{data.get('verify_count', 0)} 次")

        # 视频中找人模式：显示识别到的人员
        elif mode == "视频中找人" and data.get("stats"):
            st.markdown("---")
            st.markdown("### 👥 识别到的人员")

            stats = data.get("stats", {})
            known_persons = [name for name in stats.keys() if name != "Unknown"]

            if known_persons:
                st.write("**已识别人员:**")
                cols = st.columns(5)
                for idx, name in enumerate(sorted(known_persons)):
                    with cols[idx % 5]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 0.8rem; border-radius: 10px;
                                    text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <p style="color: white; margin: 0; font-size: 1rem;">
                                👤 {name}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            if "Unknown" in stats and stats["Unknown"] >= 5:
                st.info(f"⚠️ 已检测到陌生人 ({stats['Unknown']} 次)")

            # 显示统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总帧数", data.get("total_frames", 0))
            with col2:
                st.metric("已识别帧", data.get("processed_frames", 0))
            with col3:
                st.metric("识别人数", len(known_persons))

        else:
            st.info("⚪ 摄像头未启动 — 点击上方按钮启动，识别画面将在独立弹窗显示")

    if st.button("🔄 手动刷新", width="stretch"):
        st.rerun()


def main():
    """主函数"""
    init_session_state()
    show_sidebar()

    # 主页面导航
    page = st.sidebar.radio(
        "📍 导航",
        ["🏠 主页", "👥 人员管理", "📷 摄像头识别", "🖼️ 图片识别", "🎬 视频识别"]
    )

    if page == "🏠 主页":
        page_home()
    elif page == "👥 人员管理":
        page_person_management()
    elif page == "📷 摄像头识别":
        page_camera()
    elif page == "🖼️ 图片识别":
        page_image_recognition()
    elif page == "🎬 视频识别":
        page_video_recognition()


if __name__ == "__main__":
    main()
