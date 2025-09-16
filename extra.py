with im1:
                            st.image(
                                item.url,
                                caption=f"Original Output {i}",
                                width="stretch",
                            )
                            buf_o = io.BytesIO()
                            img.save(buf_o, format="PNG")
                            st.download_button(
                                label=f"Download Original",
                                data=buf_o.getvalue(),
                                file_name=f"napkin_original_{i}.png",
                                mime="image/png",
                                width="stretch",
                            )
                        with im2:
                            st.image(
                                enhanced_img_l,
                                caption=f"Enhanced Output {i}",
                                width="stretch",
                            )
                            buf_l = io.BytesIO()
                            enhanced_img_l.save(buf_l, format="PNG")
                            st.download_button(
                                label=f"Download Image",
                                data=buf_l.getvalue(),
                                file_name=f"napkin_{i}.png",
                                mime="image/png",
                                width="stretch",
                            )
                        with im3:
                            st.image(
                                enhanced_img_m,
                                caption=f"Enhanced (Medium) {i}",
                                width="stretch",
                            )
                            buf_m = io.BytesIO()
                            enhanced_img_m.save(buf_m, format="PNG")
                            st.download_button(
                                label=f"Download Medium",
                                data=buf_m.getvalue(),
                                file_name=f"napkin_medium_{i}.png",
                                mime="image/png",
                                width="stretch",
                            )
                        with im4:
                            st.image(
                                enhanced_img_h,
                                caption=f"Enhanced (Strong) {i}",
                                width="stretch",
                            )
                            buf_h = io.BytesIO()
                            enhanced_img_h.save(buf_h, format="PNG")
                            st.download_button(
                                label=f"Download Medium",
                                data=buf_h.getvalue(),
                                file_name=f"napkin_medium_{i}.png",
                                mime="image/png",
                                width="stretch",
                            )