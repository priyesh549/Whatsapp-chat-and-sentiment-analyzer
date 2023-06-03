# matplotlib.pyplot is imported for data visualization.
import matplotlib.pyplot as plt

import pandas as pd

# streamlit is imported as the web framework for building the application
import streamlit as st

# seaborn is imported for additional data visualization capabilities.
import seaborn as sns

import mplcursors

import plotly.graph_objects as go


# elper and preprocessor are custom modules or scripts that contain helper functions for data preprocessing and analysis.
import helper
import preprocessor

# Sidebar and main screen text and title.
st.title("WhatsApp Data Sentiment Analyzer")
st.markdown("This app is use to analyze your WhatsApp Chat using the exported text file 📁.")

st.sidebar.title("Analyze:")
st.sidebar.markdown("This app is use to analyze your WhatsApp Chat using the exported text file 📁.")

st.sidebar.markdown('*How to export chat text file?*')
st.sidebar.text('Follow the steps 👇:')
st.sidebar.text('1) Open the individual or group chat.')
st.sidebar.text('2) Tap options > More > Export chat.')
st.sidebar.text('3) Choose export without media.')

st.sidebar.markdown('You are all set to go 😃.')

st.text("Select your Exported Whatsapp Chat file....")
uploaded_file = st.sidebar.file_uploader("Choose a file")
st.sidebar.markdown("*Don't worry your data is not stored!*")
st.sidebar.markdown("*feel free to use 😊.*")
st.title("")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open("sample.txt", "wb") as binary_file:
        binary_file.write(bytes_data)
    df=preprocessor.prepro()
    # st.dataframe(df)
    # fetch unique users
    user_list = df['user'].unique().tolist()

    # Check if there are more than 2 users in the chat
    if len(user_list) > 2:
        # Remove the 'group_notification' user from the list
        user_list.remove('group_notification')


    user_list.sort()
    user_list.insert(0, "Overall")

    st.text("Select user for analysis....")
    selected_user = st.sidebar.selectbox("Show analysis w.r.t", user_list)
    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='white')
        ax.set_facecolor("black")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='white')
        ax.set_facecolor("black")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='white')
            ax.set_facecolor("black")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='white')
            ax.set_facecolor("black")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        # activity_heatmap = helper.activity_heatmap(selected_user, df)
        # fig, ax = plt.subplots()
        # ax = sns.heatmap(activity_heatmap)
        # st.pyplot(fig)

        activity_heatmap = helper.activity_heatmap(selected_user, df)

        # Define the desired order for rows and columns
        sorted_rows = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sorted_columns = ['Morning', 'Afternoon', 'Evening', 'Night']

        # Sort the rows and columns based on the desired order
        activity_heatmap = activity_heatmap.reindex(sorted_rows, axis=0)
        activity_heatmap = activity_heatmap.reindex(sorted_columns, axis=1)

        fig, ax = plt.subplots()
        ax = sns.heatmap(activity_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='white')
                ax.set_facecolor("black")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        st.title('Most commmon words')
        most_common_df = helper.most_common_words(selected_user,df)
        st.dataframe(most_common_df)



        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:

            top_emoji = emoji_df.head(5)

            fig = go.Figure(data=[go.Pie(labels=top_emoji['Emojis'], values=top_emoji['Occurences'])])

            # Add hover effect
            cursor = mplcursors.cursor(hover=True)

            @cursor.connect("add")
            def on_hover(sel):
                # Get the index of the hovered wedge
                index = sel.target.index

                # Get the corresponding emoji
                emoji = emoji_df['emoji'][index]

                # Set the tooltip text to the emoji
                sel.annotation.set_text(emoji)

            # Display the plot
            st.plotly_chart(fig)


        # Sentiment analysis
        st.title('Sentiment analysis')
        most_common_df = helper.sentiment(selected_user,df)
        st.dataframe(most_common_df)

        # Analysis
        st.title('Conclusion')
        most_common_df = helper.conclusion(selected_user,df)
        st.dataframe(most_common_df)